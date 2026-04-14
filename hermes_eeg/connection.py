# reusable_lib/eeg/connection.py
"""
BrainFlow connection manager for OpenBCI boards.
Handles board connection, streaming, and data retrieval.

Supports:
- Cyton (8-channel, 250Hz) - Full music-emotion coverage
- Cyton+Daisy (16-channel, 125Hz) - Research-grade
- Ganglion (4-channel, 200Hz) - Budget option, covers F3/F4/T7/T8
- Synthetic (8-channel, 250Hz) - Testing without hardware
- Mock (8-channel, 250Hz) - Software simulation when brainflow unavailable

Usage:
    from reusable_lib.eeg import EEGConnection

    conn = EEGConnection()

    # Connect (use 'synthetic' or 'mock' for testing without hardware)
    result = conn.connect('', 'synthetic')

    # Start streaming
    conn.start_stream()

    # Get data
    data = conn.get_current_data(250)  # 1 second at 250Hz

    # Stop and disconnect
    conn.stop_stream()
    conn.disconnect()
"""

import numpy as np
from typing import Optional, Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)

# Try to import brainflow - may fail on ARM64 due to x86 libraries
BRAINFLOW_AVAILABLE = False
BoardShim = None
BrainFlowInputParams = None
BoardIds = None

try:
    from brainflow.board_shim import BoardShim as _BoardShim
    from brainflow.board_shim import BrainFlowInputParams as _BrainFlowInputParams
    from brainflow.board_shim import BoardIds as _BoardIds
    # Test if native library actually works
    _test_params = _BrainFlowInputParams()
    BoardShim = _BoardShim
    BrainFlowInputParams = _BrainFlowInputParams
    BoardIds = _BoardIds
    BRAINFLOW_AVAILABLE = True
    logger.info("BrainFlow native library loaded successfully")
except Exception as e:
    logger.warning(f"BrainFlow not available (will use mock mode): {e}")


class MockBoard:
    """Mock board for testing when BrainFlow native library is unavailable"""

    def __init__(self, sampling_rate: int = 250, num_channels: int = 8):
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self._streaming = False
        self._start_time = None
        self._buffer = []

    def prepare_session(self):
        pass

    def start_stream(self, buffer_size: int = 45000):
        self._streaming = True
        self._start_time = time.time()
        logger.info("Mock board: streaming started")

    def stop_stream(self):
        self._streaming = False
        logger.info("Mock board: streaming stopped")

    def release_session(self):
        self._streaming = False

    def get_current_board_data(self, num_samples: int) -> np.ndarray:
        """Generate synthetic EEG-like data with realistic patterns"""
        if not self._streaming:
            return None

        t = np.linspace(0, num_samples / self.sampling_rate, num_samples)

        # Create data array: channels + timestamp
        data = np.zeros((self.num_channels + 1, num_samples))

        for ch in range(self.num_channels):
            # Mix of frequency bands to simulate real EEG
            # Alpha (10 Hz) - dominant
            alpha = 20 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
            # Beta (20 Hz)
            beta = 10 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
            # Theta (6 Hz)
            theta = 15 * np.sin(2 * np.pi * 6 * t + np.random.random() * 2 * np.pi)
            # Gamma (40 Hz)
            gamma = 5 * np.sin(2 * np.pi * 40 * t + np.random.random() * 2 * np.pi)
            # Noise
            noise = 3 * np.random.randn(num_samples)

            # Combine with some variation per channel
            data[ch] = alpha + beta * (0.8 + 0.4 * np.random.random()) + theta + gamma + noise

        # Timestamp channel
        data[-1] = np.arange(num_samples)

        return data

    def get_board_data(self, num_samples: Optional[int] = None) -> np.ndarray:
        return self.get_current_board_data(num_samples or self.sampling_rate)


class EEGConnection:
    """Manages connection to OpenBCI boards via BrainFlow (or mock when unavailable)"""

    # Board IDs - will be populated if brainflow is available
    BOARD_IDS = {}
    if BRAINFLOW_AVAILABLE and BoardIds:
        BOARD_IDS = {
            "cyton": BoardIds.CYTON_BOARD,
            "cyton_daisy": BoardIds.CYTON_DAISY_BOARD,
            "ganglion": BoardIds.GANGLION_BOARD,
            "synthetic": BoardIds.SYNTHETIC_BOARD
        }

    # Channel names for different board configurations
    CHANNEL_NAMES = {
        "cyton": ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"],
        "ganglion": ["F3", "F4", "T7", "T8"],  # 4-channel covers essentials
        "synthetic": ["Fp1", "Fp2", "F3", "F4", "T7", "T8", "P3", "P4"],
    }

    def __init__(self):
        self.board = None  # Can be BoardShim or MockBoard
        self.board_id: Optional[int] = None
        self.board_type: str = ""
        self.is_streaming: bool = False
        self.sampling_rate: int = 250
        self.eeg_channels: List[int] = []
        self.channel_names: List[str] = []
        self.is_mock: bool = False

    def connect(self, serial_port: str, board_type: str = "cyton") -> Dict[str, Any]:
        """
        Connect to the OpenBCI board (or mock if BrainFlow unavailable).

        Args:
            serial_port: Serial port (e.g., '/dev/ttyUSB0', 'COM3', or '' for synthetic/mock)
            board_type: 'cyton', 'cyton_daisy', 'ganglion', 'synthetic', or 'mock'

        Returns:
            Connection status with board info
        """
        # Force mock mode if brainflow unavailable or explicitly requested
        use_mock = (board_type == "mock") or not BRAINFLOW_AVAILABLE

        if use_mock:
            return self._connect_mock(board_type)
        else:
            return self._connect_brainflow(serial_port, board_type)

    def _connect_mock(self, board_type: str) -> Dict[str, Any]:
        """Connect using mock board (software simulation)"""
        try:
            num_channels = 4 if board_type == "ganglion" else 8
            self.board = MockBoard(sampling_rate=250, num_channels=num_channels)
            self.board_type = "mock"
            self.is_mock = True
            self.sampling_rate = 250
            self.eeg_channels = list(range(num_channels))
            self.channel_names = self.CHANNEL_NAMES.get(
                "ganglion" if num_channels == 4 else "cyton",
                [f"Ch{i+1}" for i in range(num_channels)]
            )

            self.board.prepare_session()

            logger.info(f"Connected to mock board ({num_channels} channels)")

            return {
                "success": True,
                "board_type": "mock",
                "sampling_rate": self.sampling_rate,
                "channels": num_channels,
                "channel_names": self.channel_names,
                "message": f"Connected to mock board (BrainFlow unavailable on this platform)",
                "note": "Using software-simulated EEG data for testing"
            }

        except Exception as e:
            logger.error(f"Mock connection failed: {e}")
            return {"success": False, "error": str(e)}

    def _connect_brainflow(self, serial_port: str, board_type: str) -> Dict[str, Any]:
        """Connect using real BrainFlow board"""
        try:
            # Enable BrainFlow logging for debugging
            BoardShim.enable_dev_board_logger()

            params = BrainFlowInputParams()
            if serial_port:  # Empty string for synthetic board
                params.serial_port = serial_port

            self.board_id = self.BOARD_IDS.get(board_type, BoardIds.CYTON_BOARD)
            self.board_type = board_type
            self.is_mock = False
            self.board = BoardShim(self.board_id, params)

            self.board.prepare_session()

            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.channel_names = self.CHANNEL_NAMES.get(
                board_type,
                [f"Ch{i+1}" for i in range(len(self.eeg_channels))]
            )

            logger.info(f"Connected to {board_type} on {serial_port or 'synthetic'}")

            return {
                "success": True,
                "board_type": board_type,
                "sampling_rate": self.sampling_rate,
                "channels": len(self.eeg_channels),
                "channel_names": self.channel_names,
                "message": f"Connected to {board_type}" + (f" on {serial_port}" if serial_port else " (synthetic)")
            }

        except Exception as e:
            logger.error(f"BrainFlow connection failed: {e}")
            # Fall back to mock mode
            logger.info("Falling back to mock mode")
            return self._connect_mock(board_type)

    def start_stream(self) -> Dict[str, Any]:
        """Start data streaming from the board"""
        if not self.board:
            return {"success": False, "error": "Not connected"}

        try:
            self.board.start_stream(45000)  # Ring buffer size
            self.is_streaming = True
            logger.info("EEG streaming started")
            return {"success": True, "message": "Streaming started"}
        except Exception as e:
            logger.error(f"Stream start failed: {e}")
            return {"success": False, "error": str(e)}

    def stop_stream(self) -> Dict[str, Any]:
        """Stop data streaming"""
        if self.board and self.is_streaming:
            try:
                self.board.stop_stream()
                self.is_streaming = False
                logger.info("EEG streaming stopped")
                return {"success": True, "message": "Streaming stopped"}
            except Exception as e:
                logger.error(f"Stream stop failed: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Not streaming"}

    def get_current_data(self, num_samples: int = 250) -> Optional[np.ndarray]:
        """
        Get current EEG data from the ring buffer.

        Args:
            num_samples: Number of samples to retrieve (default: 1 second at 250Hz)

        Returns:
            numpy array of shape (channels, samples) or None if not streaming
        """
        if not self.board or not self.is_streaming:
            return None
        try:
            return self.board.get_current_board_data(num_samples)
        except Exception as e:
            logger.error(f"Failed to get data: {e}")
            return None

    def get_board_data(self, num_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get and remove data from the ring buffer.

        Args:
            num_samples: Number of samples to retrieve (None = all available)

        Returns:
            numpy array of shape (channels, samples) or None
        """
        if not self.board or not self.is_streaming:
            return None
        try:
            if num_samples:
                return self.board.get_board_data(num_samples)
            else:
                return self.board.get_board_data()
        except Exception as e:
            logger.error(f"Failed to get data: {e}")
            return None

    def disconnect(self) -> Dict[str, Any]:
        """Disconnect from the board and release resources"""
        try:
            if self.board:
                if self.is_streaming:
                    self.stop_stream()
                self.board.release_session()
                self.board = None
                self.board_id = None
                self.is_streaming = False
                logger.info("EEG disconnected")
                return {"success": True, "message": "Disconnected"}
            return {"success": True, "message": "Already disconnected"}
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current connection and streaming status"""
        return {
            "connected": self.board is not None,
            "streaming": self.is_streaming,
            "board_type": self.board_type,
            "sampling_rate": self.sampling_rate,
            "channels": len(self.eeg_channels),
            "channel_names": self.channel_names
        }
