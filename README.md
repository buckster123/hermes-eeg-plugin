# hermes-eeg 🧠

EEG/BCI neural interface plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). Connects OpenBCI hardware to AI agents via real-time emotion detection and AI-readable "felt experience" generation.

**The core innovation:** AI agents can perceive how humans *experience* music, content, or any stimulus — enabling a closed feedback loop between human emotion and AI creation.

## Features

- **OpenBCI Hardware Support** — Cyton (8-ch), Ganglion (4-ch), Synthetic (test data)
- **Works Without Hardware** — Mock mode with realistic simulated EEG for development/testing
- **Real-time Emotion Detection** — Valence, arousal, attention, engagement at 2Hz
- **Musical Chills Detection** — Gamma bursts + theta coupling = frisson events
- **Felt Experience Format** — AI-readable session data with emotional arcs and narratives
- **Session Recording** — Persistent storage of listening sessions as JSON

## Installation

```bash
# Core (mock mode, no hardware needed)
pip install hermes-eeg

# With OpenBCI hardware support
pip install "hermes-eeg[hardware]"

# Enable the toolset in Hermes
hermes tools enable eeg
```

## Tools

| Tool | Description |
|------|-------------|
| `eeg_connect` | Connect to OpenBCI board (or mock/synthetic) |
| `eeg_disconnect` | Disconnect and release resources |
| `eeg_stream_start` | Start recording a listening session |
| `eeg_stream_stop` | Stop and generate felt experience format |
| `eeg_realtime_emotion` | Get live emotional state |
| `eeg_experience_get` | Retrieve past session data |
| `eeg_calibrate_baseline` | Prepare personal baseline calibration |
| `eeg_list_sessions` | Browse recorded sessions |

## Quick Start

```
# In Hermes chat:
> Connect to EEG in mock mode and start a test session

# The agent will:
# 1. eeg_connect(serial_port="", board_type="mock")
# 2. eeg_stream_start(session_name="test", track_title="My Song")
# 3. ... record emotional data in background ...
# 4. eeg_stream_stop() → generates narrative + summary
```

## How It Works

### Signal Processing Pipeline
1. **Raw EEG** → Detrend → Notch filter (50/60Hz) → Bandpass (0.5-45Hz)
2. **Band Power Extraction** (Welch's method):
   - Theta (4-8 Hz) — Emotional processing
   - Alpha (8-13 Hz) — Relaxation
   - Beta (13-30 Hz) — Arousal
   - Gamma (30-45 Hz) — Peak experience / "chills"
3. **Emotion Mapping**:
   - **Valence** (-1 to +1): Frontal alpha asymmetry (F4-F3)
   - **Arousal** (0-1): Beta/alpha ratio
   - **Attention** (0-1): Theta/beta ratio + gamma
   - **Engagement** (0-1): Geometric mean of arousal × attention

### Felt Experience Format
Sessions are saved as JSON with:
- Per-moment emotional dimensions (valence, arousal, attention, engagement)
- Event flags (attention_shift, emotional_peak, possible_chills)
- Summary statistics
- Natural language narrative for AI consumption

## Dependencies

- **numpy** + **scipy** — Always required (signal processing)
- **brainflow** — Optional (real OpenBCI hardware). Falls back to mock/SciPy without it.

## Development

```bash
git clone https://github.com/buckster123/hermes-eeg-plugin
cd hermes-eeg-plugin
pip install -e ".[dev]"
pytest tests/ -v
```

## Based On

Extracted from [ApexAurum](https://github.com/buckster123/ApexAurum) — the neural resonance system for AI-human creative feedback loops.

## License

MIT
