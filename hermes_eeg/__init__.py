# hermes_eeg/__init__.py
"""
Hermes EEG Plugin — Neural Interface for AI Agents

Connects OpenBCI EEG hardware (or mock/synthetic boards) to Hermes Agent,
enabling real-time emotion detection and AI-readable "felt experience" generation.

Tools:
  eeg_connect          - Connect to OpenBCI board (Cyton/Ganglion/synthetic/mock)
  eeg_disconnect       - Disconnect and release resources
  eeg_stream_start     - Start streaming + recording a listening session
  eeg_stream_stop      - Stop streaming, generate experience format
  eeg_realtime_emotion - Get current emotional state (live)
  eeg_experience_get   - Retrieve past session experience data
  eeg_calibrate_baseline - Prepare baseline calibration
  eeg_list_sessions    - List recorded sessions

Works without hardware: uses mock board + SciPy signal processing when
brainflow is not installed. Install brainflow for real hardware support.

Requirements: numpy, scipy (always). brainflow (optional, for real hardware).
"""

import json
import os
import time
import threading
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# State management (lazy singleton)
# =============================================================================

_eeg_manager = None
_stream_thread = None
_session_moments = []


def _get_data_dir() -> Path:
    """Get plugin data directory, respecting Hermes profiles."""
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = Path.home() / ".hermes"
    data_dir = base / "eeg"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sessions").mkdir(exist_ok=True)
    return data_dir


def _get_eeg_manager():
    """Lazy initialization of EEG manager components."""
    global _eeg_manager
    if _eeg_manager is None:
        from .connection import EEGConnection
        from .processor import EEGProcessor
        from .experience import EmotionMapper, ListeningSession, MomentExperience

        _eeg_manager = {
            'connection': EEGConnection(),
            'processor': EEGProcessor(),
            'mapper': EmotionMapper(),
            'ListeningSession': ListeningSession,
            'MomentExperience': MomentExperience,
            'current_session': None,
            'session_start_time': None,
        }
    return _eeg_manager


def _format_timestamp(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    seconds = ms // 1000
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"


def _interpret_emotion(valence: float, arousal: float) -> str:
    """Generate human-readable interpretation of emotional state."""
    if valence > 0.4 and arousal > 0.6:
        return "Joyful/Excited"
    elif valence > 0.4 and arousal > 0.3:
        return "Happy/Content"
    elif valence > 0.4:
        return "Calm/Peaceful"
    elif valence < -0.2 and arousal > 0.6:
        return "Tense/Agitated"
    elif valence < -0.2:
        return "Sad/Melancholic"
    elif arousal > 0.6:
        return "Alert/Engaged"
    else:
        return "Neutral/Relaxed"


# =============================================================================
# Tool Handlers (must return JSON strings)
# =============================================================================

def _handle_eeg_connect(args: dict, **kw) -> str:
    serial_port = args.get("serial_port", "")
    board_type = args.get("board_type", "cyton")
    try:
        mgr = _get_eeg_manager()
        result = mgr['connection'].connect(serial_port, board_type)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_disconnect(args: dict, **kw) -> str:
    try:
        mgr = _get_eeg_manager()
        return json.dumps(mgr['connection'].disconnect())
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_stream_start(args: dict, **kw) -> str:
    global _session_moments, _stream_thread

    session_name = args.get("session_name", "unnamed")
    track_id = args.get("track_id", "")
    track_title = args.get("track_title", "")
    listener_name = args.get("listener_name", "User")

    try:
        mgr = _get_eeg_manager()

        if not mgr['connection'].board:
            return json.dumps({"success": False, "error": "Not connected. Call eeg_connect first."})

        # Start streaming
        result = mgr['connection'].start_stream()
        if not result.get("success"):
            return json.dumps(result)

        # Create session
        session_id = f"listen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
        _session_moments = []

        mgr['current_session'] = {
            'id': session_id,
            'name': session_name,
            'track_id': track_id,
            'track_title': track_title or session_name,
            'listener': listener_name,
            'start_time': datetime.now(),
        }
        mgr['session_start_time'] = time.time()

        # Background processing thread
        def process_stream():
            conn = mgr['connection']
            proc = mgr['processor']
            mapper = mgr['mapper']
            start_time = time.time()

            while conn.is_streaming:
                try:
                    data = conn.get_current_data(conn.sampling_rate)
                    if data is not None and data.shape[1] >= conn.sampling_rate // 2:
                        processed = proc.process_window(
                            data, conn.eeg_channels, conn.channel_names
                        )
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        moment = mapper.process_moment(
                            processed['band_powers'],
                            timestamp_ms=elapsed_ms,
                            track_position=_format_timestamp(elapsed_ms),
                        )
                        _session_moments.append(moment)
                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Stream processing warning: {e}")
                    time.sleep(0.5)

        _stream_thread = threading.Thread(target=process_stream, daemon=True)
        _stream_thread.start()

        return json.dumps({
            "success": True,
            "session_id": session_id,
            "track_id": track_id,
            "track_title": track_title or session_name,
            "listener": listener_name,
            "message": f"Streaming started for session: {session_name}",
            "board_info": mgr['connection'].get_status(),
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_stream_stop(args: dict, **kw) -> str:
    global _session_moments

    generate_experience = args.get("generate_experience", True)
    try:
        mgr = _get_eeg_manager()
        mgr['connection'].stop_stream()

        session = mgr['current_session']
        if not session:
            return json.dumps({"success": True, "message": "Streaming stopped (no active session)"})

        duration_ms = 0
        if mgr['session_start_time']:
            duration_ms = int((time.time() - mgr['session_start_time']) * 1000)

        result = {
            "success": True,
            "session_id": session['id'],
            "duration_ms": duration_ms,
            "duration_formatted": _format_timestamp(duration_ms),
            "moments_recorded": len(_session_moments),
        }

        if generate_experience and _session_moments:
            ListeningSession = mgr['ListeningSession']
            listening_session = ListeningSession(
                session_id=session['id'],
                track_id=session['track_id'],
                track_title=session['track_title'],
                listener=session['listener'],
                duration_ms=duration_ms,
                moments=_session_moments,
            )

            sessions_dir = _get_data_dir() / "sessions"
            filepath = sessions_dir / f"{session['id']}.json"
            listening_session.save_to_file(str(filepath))

            experience_data = listening_session.to_dict()
            result["experience"] = experience_data
            result["narrative"] = experience_data.get("experience_narrative", "")
            result["summary"] = experience_data.get("summary", {})
            result["saved_to"] = str(filepath)

        # Clear session
        mgr['current_session'] = None
        mgr['session_start_time'] = None
        _session_moments = []

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_realtime_emotion(args: dict, **kw) -> str:
    try:
        mgr = _get_eeg_manager()
        if not mgr['connection'].is_streaming:
            return json.dumps({"success": False, "error": "Not streaming. Call eeg_stream_start first."})

        conn = mgr['connection']
        data = conn.get_current_data(conn.sampling_rate)
        if data is None or data.shape[1] < conn.sampling_rate // 2:
            return json.dumps({"success": False, "error": "Insufficient data. Wait a moment and try again."})

        processed = mgr['processor'].process_window(data, conn.eeg_channels, conn.channel_names)
        moment = mgr['mapper'].process_moment(
            processed['band_powers'], timestamp_ms=0, track_position="live", include_raw=False
        )

        return json.dumps({
            "success": True,
            "valence": round(moment.valence, 3),
            "arousal": round(moment.arousal, 3),
            "attention": round(moment.attention, 3),
            "engagement": round(moment.engagement, 3),
            "possible_chills": moment.possible_chills,
            "emotional_peak": moment.emotional_peak,
            "interpretation": _interpret_emotion(moment.valence, moment.arousal),
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_experience_get(args: dict, **kw) -> str:
    session_id = args.get("session_id", "")
    detail_level = args.get("detail_level", "full")

    try:
        sessions_dir = _get_data_dir() / "sessions"
        filepath = sessions_dir / f"{session_id}.json"

        if not filepath.exists():
            available = [f.stem for f in sessions_dir.glob("*.json")]
            return json.dumps({
                "success": False,
                "error": f"Session not found: {session_id}",
                "available_sessions": available[:10],
            })

        with open(filepath) as f:
            data = json.load(f)

        if detail_level == "summary":
            return json.dumps({
                "success": True,
                "session_id": session_id,
                "track_title": data.get("track_title", ""),
                "listener": data.get("listener", ""),
                "duration_ms": data.get("duration_ms", 0),
                "summary": data.get("summary", {}),
                "narrative": data.get("experience_narrative", ""),
            })
        elif detail_level == "narrative":
            return json.dumps({
                "success": True,
                "session_id": session_id,
                "narrative": data.get("experience_narrative", ""),
            })
        else:
            return json.dumps({"success": True, **data})

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_calibrate_baseline(args: dict, **kw) -> str:
    listener_name = args.get("listener_name", "User")
    try:
        mgr = _get_eeg_manager()
        if not mgr['connection'].board:
            return json.dumps({"success": False, "error": "Not connected. Call eeg_connect first."})

        return json.dumps({
            "success": True,
            "message": "Baseline calibration ready",
            "listener": listener_name,
            "status": "ready_to_start",
            "instructions": [
                "1. Sit comfortably and relax",
                "2. When prompted, keep eyes OPEN for 30 seconds",
                "3. Then keep eyes CLOSED for 30 seconds",
                "4. Baseline will be saved for future sessions",
                "5. Run eeg_stream_start with session_name='calibration' to begin",
            ],
            "note": "Calibration improves emotion detection accuracy by establishing your personal baseline patterns.",
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _handle_eeg_list_sessions(args: dict, **kw) -> str:
    limit = args.get("limit", 10)
    try:
        sessions_dir = _get_data_dir() / "sessions"
        if not sessions_dir.exists():
            return json.dumps({"success": True, "sessions": [], "count": 0})

        sessions = []
        files = sorted(sessions_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        for filepath in files[:limit]:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", filepath.stem),
                    "track_title": data.get("track_title", "Unknown"),
                    "listener": data.get("listener", "Unknown"),
                    "duration_ms": data.get("duration_ms", 0),
                    "created_at": data.get("created_at", ""),
                    "chills_count": data.get("summary", {}).get("chills_count", 0),
                })
            except Exception as e:
                logger.warning(f"Failed to read session {filepath.name}: {e}")

        return json.dumps({
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "total_available": len(files),
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# =============================================================================
# Tool Schemas (OpenAI function-calling format)
# =============================================================================

TOOL_SCHEMAS = {
    "eeg_connect": {
        "name": "eeg_connect",
        "description": "Connect to OpenBCI EEG board for neural/emotion sensing. Supports Cyton (8-channel), Ganglion (4-channel), synthetic (fake data for testing), or mock (software simulation). Works without hardware — automatically falls back to mock mode if brainflow is not installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "serial_port": {
                    "type": "string",
                    "description": "Serial port (e.g., '/dev/ttyUSB0' on Linux, 'COM3' on Windows). Use empty string '' for synthetic/mock board.",
                },
                "board_type": {
                    "type": "string",
                    "enum": ["cyton", "ganglion", "synthetic", "mock"],
                    "description": "Board type: 'cyton' (8-ch, 250Hz), 'ganglion' (4-ch, 200Hz), 'synthetic' (BrainFlow test data), 'mock' (software simulation).",
                },
            },
            "required": ["serial_port"],
        },
    },
    "eeg_disconnect": {
        "name": "eeg_disconnect",
        "description": "Disconnect from the EEG board and release resources.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    "eeg_stream_start": {
        "name": "eeg_stream_start",
        "description": "Start EEG streaming and recording for a listening session. Continuously records emotional response data in the background at 2Hz. Must call eeg_connect first.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_name": {
                    "type": "string",
                    "description": "Name for this listening session",
                },
                "track_id": {
                    "type": "string",
                    "description": "ID of the track being listened to (e.g., from music_generate)",
                },
                "track_title": {
                    "type": "string",
                    "description": "Title of the track",
                },
                "listener_name": {
                    "type": "string",
                    "description": "Name of the listener",
                },
            },
            "required": ["session_name"],
        },
    },
    "eeg_stream_stop": {
        "name": "eeg_stream_stop",
        "description": "Stop EEG streaming and generate the AI-readable 'felt experience' format. Returns emotional summary, narrative, and saves session to disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "generate_experience": {
                    "type": "boolean",
                    "description": "Generate and save the felt experience format (default: true)",
                },
            },
        },
    },
    "eeg_realtime_emotion": {
        "name": "eeg_realtime_emotion",
        "description": "Get current real-time emotional state during active EEG streaming. Returns valence (-1 to +1), arousal (0-1), attention (0-1), engagement (0-1), and detects musical chills.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    "eeg_experience_get": {
        "name": "eeg_experience_get",
        "description": "Retrieve the felt experience from a recorded listening session. This is how AI agents 'feel' what the human experienced during music listening — emotional arc, peak moments, chills, and narrative.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to retrieve (e.g., 'listen_20260414_143022_123')",
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["summary", "full", "narrative"],
                    "description": "Detail level: 'summary' (stats + narrative), 'full' (all moment data), 'narrative' (just natural language description)",
                },
            },
            "required": ["session_id"],
        },
    },
    "eeg_calibrate_baseline": {
        "name": "eeg_calibrate_baseline",
        "description": "Prepare for baseline EEG calibration. Records personal resting-state patterns to improve emotion detection accuracy.",
        "parameters": {
            "type": "object",
            "properties": {
                "listener_name": {
                    "type": "string",
                    "description": "Name of the person being calibrated",
                },
            },
        },
    },
    "eeg_list_sessions": {
        "name": "eeg_list_sessions",
        "description": "List recorded EEG listening sessions with metadata (track, listener, duration, chills count).",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum sessions to return (default: 10)",
                },
            },
        },
    },
}

TOOL_HANDLERS = {
    "eeg_connect": _handle_eeg_connect,
    "eeg_disconnect": _handle_eeg_disconnect,
    "eeg_stream_start": _handle_eeg_stream_start,
    "eeg_stream_stop": _handle_eeg_stream_stop,
    "eeg_realtime_emotion": _handle_eeg_realtime_emotion,
    "eeg_experience_get": _handle_eeg_experience_get,
    "eeg_calibrate_baseline": _handle_eeg_calibrate_baseline,
    "eeg_list_sessions": _handle_eeg_list_sessions,
}


# =============================================================================
# Check functions
# =============================================================================

def _check_eeg_available() -> bool:
    """EEG tools are always available — mock mode works without any deps beyond numpy/scipy."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Plugin registration — called by Hermes PluginManager
# =============================================================================

def register(ctx):
    """Called by Hermes PluginManager on startup."""
    toolset = "eeg"

    for name, schema in TOOL_SCHEMAS.items():
        ctx.register_tool(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=TOOL_HANDLERS[name],
            check_fn=_check_eeg_available,
            requires_env=[],
            emoji="🧠",
        )

    logger.info(f"hermes-eeg: registered {len(TOOL_SCHEMAS)} tools in toolset '{toolset}'")
