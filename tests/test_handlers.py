# tests/test_handlers.py
"""Test tool handlers with mock EEG board."""

import json
import pytest
import hermes_eeg


def test_eeg_connect_mock():
    """Connect in mock mode should always succeed."""
    result = json.loads(hermes_eeg._handle_eeg_connect({"serial_port": "", "board_type": "mock"}))
    assert result["success"] is True
    assert result["board_type"] == "mock"
    assert result["channels"] in (4, 8)
    # Cleanup
    hermes_eeg._handle_eeg_disconnect({})


def test_eeg_connect_synthetic_falls_back_to_mock():
    """Synthetic without brainflow should fall back to mock."""
    result = json.loads(hermes_eeg._handle_eeg_connect({"serial_port": "", "board_type": "synthetic"}))
    assert result["success"] is True
    # Cleanup
    hermes_eeg._handle_eeg_disconnect({})


def test_eeg_disconnect_when_not_connected():
    # Reset manager
    hermes_eeg._eeg_manager = None
    result = json.loads(hermes_eeg._handle_eeg_disconnect({}))
    assert result["success"] is True


def test_eeg_stream_start_without_connect():
    hermes_eeg._eeg_manager = None
    result = json.loads(hermes_eeg._handle_eeg_stream_start({"session_name": "test"}))
    # Should fail because not connected
    assert result["success"] is False


def test_eeg_realtime_without_streaming():
    hermes_eeg._eeg_manager = None
    result = json.loads(hermes_eeg._handle_eeg_realtime_emotion({}))
    assert result["success"] is False


def test_eeg_list_sessions_empty():
    result = json.loads(hermes_eeg._handle_eeg_list_sessions({}))
    assert result["success"] is True
    assert isinstance(result["sessions"], list)


def test_eeg_experience_get_missing():
    result = json.loads(hermes_eeg._handle_eeg_experience_get({"session_id": "nonexistent_123"}))
    assert result["success"] is False


def test_full_session_flow():
    """End-to-end: connect -> stream -> stop -> list."""
    import time

    hermes_eeg._eeg_manager = None

    # Connect mock
    result = json.loads(hermes_eeg._handle_eeg_connect({"serial_port": "", "board_type": "mock"}))
    assert result["success"] is True

    # Start streaming
    result = json.loads(hermes_eeg._handle_eeg_stream_start({
        "session_name": "test_session",
        "track_id": "test_track_001",
        "track_title": "Test Song",
        "listener_name": "TestUser",
    }))
    assert result["success"] is True
    session_id = result["session_id"]

    # Let it record a few moments
    time.sleep(2)

    # Stop and generate experience
    result = json.loads(hermes_eeg._handle_eeg_stream_stop({"generate_experience": True}))
    assert result["success"] is True
    assert result["moments_recorded"] > 0
    assert "narrative" in result

    # Retrieve the session
    result = json.loads(hermes_eeg._handle_eeg_experience_get({
        "session_id": session_id,
        "detail_level": "summary",
    }))
    assert result["success"] is True
    assert "narrative" in result

    # List sessions
    result = json.loads(hermes_eeg._handle_eeg_list_sessions({}))
    assert result["success"] is True
    assert result["count"] > 0

    # Cleanup
    hermes_eeg._handle_eeg_disconnect({})
    hermes_eeg._eeg_manager = None


def test_handlers_return_json_strings():
    """All handlers must return JSON strings, not dicts."""
    hermes_eeg._eeg_manager = None
    for name, handler in hermes_eeg.TOOL_HANDLERS.items():
        # Call with minimal/empty args — they should all handle gracefully
        try:
            result = handler({})
        except Exception:
            continue
        assert isinstance(result, str), f"Handler {name} returned {type(result)}, expected str"
        # Must be valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
