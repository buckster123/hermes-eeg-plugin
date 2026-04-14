# tests/test_processing.py
"""Test EEG signal processing and emotion mapping."""

import numpy as np
import json
import pytest
from hermes_eeg.connection import MockBoard
from hermes_eeg.processor import EEGProcessor, BandPower
from hermes_eeg.experience import EmotionMapper, MomentExperience, ListeningSession


class TestMockBoard:
    def test_creates_with_defaults(self):
        board = MockBoard()
        assert board.sampling_rate == 250
        assert board.num_channels == 8

    def test_generates_data_when_streaming(self):
        board = MockBoard()
        board.prepare_session()
        board.start_stream()
        data = board.get_current_board_data(250)
        assert data is not None
        assert data.shape == (9, 250)  # 8 channels + 1 timestamp
        board.stop_stream()

    def test_returns_none_when_not_streaming(self):
        board = MockBoard()
        board.prepare_session()
        assert board.get_current_board_data(250) is None

    def test_ganglion_channel_count(self):
        board = MockBoard(num_channels=4)
        board.prepare_session()
        board.start_stream()
        data = board.get_current_board_data(200)
        assert data.shape[0] == 5  # 4 channels + 1 timestamp
        board.stop_stream()


class TestEEGProcessor:
    def setup_method(self):
        self.processor = EEGProcessor(sampling_rate=250, num_channels=8)
        self.board = MockBoard()
        self.board.prepare_session()
        self.board.start_stream()

    def teardown_method(self):
        self.board.stop_stream()

    def test_preprocess_returns_array(self):
        data = self.board.get_current_board_data(250)
        cleaned = self.processor.preprocess(data, 0)
        assert isinstance(cleaned, np.ndarray)
        assert len(cleaned) == 250

    def test_extract_band_powers(self):
        data = self.board.get_current_board_data(250)
        cleaned = self.processor.preprocess(data, 0)
        powers = self.processor.extract_band_powers(cleaned)
        assert isinstance(powers, BandPower)
        assert powers.theta >= 0
        assert powers.alpha >= 0
        assert powers.beta >= 0
        assert powers.gamma >= 0

    def test_band_power_to_dict(self):
        bp = BandPower(theta=1.234567, alpha=2.345678, beta=3.456789, gamma=4.567890)
        d = bp.to_dict()
        assert d["theta"] == 1.2346
        assert d["alpha"] == 2.3457

    def test_process_window(self):
        data = self.board.get_current_board_data(250)
        channels = list(range(8))
        names = ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"]
        results = self.processor.process_window(data, channels, names)
        assert "band_powers" in results
        assert "differential_entropy" in results
        assert "F3" in results["band_powers"]

    def test_differential_entropy(self):
        sig = np.random.randn(250)
        de = self.processor.calculate_differential_entropy(sig)
        assert isinstance(de, float)
        assert de > 0  # Random signal has positive DE

    def test_short_signal_handling(self):
        """Very short signals should return zero band powers."""
        short = np.random.randn(5)
        powers = self.processor.extract_band_powers(short)
        assert powers.theta == 0.0

    def test_normalize_powers(self):
        band_powers = {
            "F3": BandPower(theta=2.0, alpha=4.0, beta=1.0, gamma=0.5),
            "F4": BandPower(theta=1.0, alpha=2.0, beta=2.0, gamma=1.0),
        }
        normalized = self.processor.normalize_powers(band_powers)
        assert normalized["F3"].theta == 1.0  # Max theta
        assert normalized["F4"].theta == 0.5


class TestEmotionMapper:
    def setup_method(self):
        self.mapper = EmotionMapper()
        self.mock_powers = {
            "F3": BandPower(theta=1.0, alpha=3.0, beta=1.5, gamma=0.5),
            "F4": BandPower(theta=1.0, alpha=4.0, beta=1.5, gamma=0.5),
            "T7": BandPower(theta=0.8, alpha=2.5, beta=1.2, gamma=0.3),
            "T8": BandPower(theta=0.8, alpha=2.5, beta=1.2, gamma=0.3),
        }

    def test_valence_range(self):
        v = self.mapper.calculate_valence(self.mock_powers)
        assert -1 <= v <= 1

    def test_arousal_range(self):
        a = self.mapper.calculate_arousal(self.mock_powers)
        assert 0 <= a <= 1

    def test_attention_range(self):
        a = self.mapper.calculate_attention(self.mock_powers)
        assert 0 <= a <= 1

    def test_engagement_range(self):
        e = self.mapper.calculate_engagement(0.5, 0.5)
        assert 0 <= e <= 1

    def test_process_moment(self):
        moment = self.mapper.process_moment(self.mock_powers, timestamp_ms=1000)
        assert isinstance(moment, MomentExperience)
        assert -1 <= moment.valence <= 1
        assert 0 <= moment.arousal <= 1
        assert moment.timestamp_ms == 1000

    def test_moment_to_dict(self):
        moment = self.mapper.process_moment(self.mock_powers, timestamp_ms=1000, track_position="0:01")
        d = moment.to_dict()
        assert "valence" in d
        assert "arousal" in d
        assert d["timestamp_ms"] == 1000

    def test_detect_chills(self):
        """Chills require high gamma + theta + arousal + attention."""
        no_chills = self.mapper.detect_chills(self.mock_powers, arousal=0.5, attention=0.5)
        assert no_chills is False

        # Create powers that should trigger chills
        chills_powers = {
            "F3": BandPower(theta=1.0, alpha=0.5, beta=2.0, gamma=1.0),
            "F4": BandPower(theta=1.0, alpha=0.5, beta=2.0, gamma=1.0),
        }
        chills = self.mapper.detect_chills(chills_powers, arousal=0.8, attention=0.8)
        assert chills is True

    def test_attention_shift_detection(self):
        self.mapper.previous_attention = 0.3
        assert self.mapper.detect_attention_shift(0.8) is True   # Big shift
        assert self.mapper.detect_attention_shift(0.85) is False  # Small shift


class TestListeningSession:
    def test_empty_session(self):
        session = ListeningSession(
            session_id="test_001",
            track_id="track_001",
            track_title="Test",
            listener="User",
            duration_ms=0,
        )
        d = session.to_dict()
        assert d["session_id"] == "test_001"
        assert d["summary"]["chills_count"] == 0
        assert "No listening data" in d["experience_narrative"]

    def test_session_with_moments(self):
        moments = [
            MomentExperience(
                timestamp_ms=i * 500, track_position=f"0:{i//2:02d}",
                valence=0.5, arousal=0.6, attention=0.7, engagement=0.65,
            )
            for i in range(10)
        ]
        session = ListeningSession(
            session_id="test_002", track_id="t1", track_title="Song",
            listener="User", duration_ms=5000, moments=moments,
        )
        summary = session.generate_summary()
        assert summary["overall_valence"] > 0
        narrative = session.generate_narrative()
        assert "User" in narrative
        assert "Song" in narrative

    def test_save_and_load(self, tmp_path):
        moments = [
            MomentExperience(
                timestamp_ms=0, track_position="0:00",
                valence=0.3, arousal=0.4, attention=0.5, engagement=0.45,
            )
        ]
        session = ListeningSession(
            session_id="test_003", track_id="t1", track_title="RoundTrip",
            listener="User", duration_ms=1000, moments=moments,
        )
        filepath = tmp_path / "test_003.json"
        assert session.save_to_file(str(filepath)) is True
        loaded = ListeningSession.load_from_file(str(filepath))
        assert loaded is not None
        assert loaded.session_id == "test_003"
        assert len(loaded.moments) == 1
        assert loaded.moments[0].valence == 0.3
