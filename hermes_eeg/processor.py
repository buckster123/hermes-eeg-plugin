# reusable_lib/eeg/processor.py
"""
Real-time EEG signal processing.
Handles filtering, frequency decomposition, and feature extraction.

Based on music-emotion EEG research:
- Theta (4-8 Hz): Emotional processing, strongly linked to valence
- Alpha (8-13 Hz): Relaxation, increases with pleasant music
- Beta (13-30 Hz): Arousal and emotional intensity
- Gamma (30-45 Hz): Peak experience, musical "chills"

Supports both BrainFlow (when available) and pure NumPy/SciPy fallback.

Usage:
    from reusable_lib.eeg import EEGProcessor, BandPower

    processor = EEGProcessor(sampling_rate=250, num_channels=8)

    # Process a window of data
    results = processor.process_window(data, eeg_channels, channel_names)

    # Access band powers
    for ch_name, powers in results["band_powers"].items():
        print(f"{ch_name}: theta={powers.theta}, alpha={powers.alpha}")
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import and test brainflow data filter
BRAINFLOW_FILTER_AVAILABLE = False
DataFilter = None
FilterTypes = None
WindowOperations = None

try:
    from brainflow.data_filter import DataFilter as _DataFilter
    from brainflow.data_filter import FilterTypes as _FilterTypes
    from brainflow.data_filter import WindowOperations as _WindowOperations
    # Test if native library actually works by calling a simple function
    _test_nfft = _DataFilter.get_nearest_power_of_two(256)
    DataFilter = _DataFilter
    FilterTypes = _FilterTypes
    WindowOperations = _WindowOperations
    BRAINFLOW_FILTER_AVAILABLE = True
    logger.info("BrainFlow DataFilter loaded successfully")
except Exception as e:
    logger.info(f"Using SciPy for signal processing (BrainFlow unavailable: {type(e).__name__})")


@dataclass
class BandPower:
    """Power in each frequency band for one channel"""
    theta: float   # 4-8 Hz - Emotional processing
    alpha: float   # 8-13 Hz - Relaxation
    beta: float    # 13-30 Hz - Arousal
    gamma: float   # 30-45 Hz - Peak experience

    def to_dict(self) -> Dict[str, float]:
        return {
            "theta": round(self.theta, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "gamma": round(self.gamma, 4)
        }


class EEGProcessor:
    """Real-time EEG signal processing pipeline"""

    # Frequency band definitions (Hz)
    BANDS = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }

    def __init__(self, sampling_rate: int = 250, num_channels: int = 8):
        """
        Initialize the processor.

        Args:
            sampling_rate: Sample rate in Hz (Cyton: 250, Ganglion: 200)
            num_channels: Number of EEG channels
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels

    def preprocess(self, data: np.ndarray, channel_idx: int) -> np.ndarray:
        """
        Apply preprocessing filters to a single channel.

        Pipeline:
        1. Detrend (remove DC offset)
        2. Notch filter (remove 50/60Hz power line noise)
        3. Bandpass filter (0.5-45 Hz)

        Args:
            data: Full data array from board
            channel_idx: Index of the channel to process

        Returns:
            Cleaned signal array
        """
        sig = data[channel_idx].copy().astype(np.float64)

        if len(sig) < 10:
            return sig

        if BRAINFLOW_FILTER_AVAILABLE:
            return self._preprocess_brainflow(sig)
        else:
            return self._preprocess_scipy(sig)

    def _preprocess_brainflow(self, sig: np.ndarray) -> np.ndarray:
        """Preprocess using BrainFlow filters"""
        try:
            DataFilter.detrend(sig, 1)
            DataFilter.remove_environmental_noise(sig, self.sampling_rate, 1)
            DataFilter.perform_bandpass(
                sig, self.sampling_rate,
                0.5, 45.0, 4,
                FilterTypes.BUTTERWORTH, 0
            )
        except Exception as e:
            logger.warning(f"BrainFlow preprocessing warning: {e}")
        return sig

    def _preprocess_scipy(self, sig: np.ndarray) -> np.ndarray:
        """Preprocess using SciPy filters (fallback)"""
        try:
            # Detrend
            sig = signal.detrend(sig)

            # Notch filter at 50Hz
            b_notch, a_notch = signal.iirnotch(50, 30, self.sampling_rate)
            sig = signal.filtfilt(b_notch, a_notch, sig)

            # Bandpass 0.5-45 Hz
            nyq = self.sampling_rate / 2
            low = 0.5 / nyq
            high = 45.0 / nyq
            b_bp, a_bp = signal.butter(4, [low, high], btype='band')
            sig = signal.filtfilt(b_bp, a_bp, sig)

        except Exception as e:
            logger.warning(f"SciPy preprocessing warning: {e}")

        return sig

    def extract_band_powers(self, sig: np.ndarray) -> BandPower:
        """
        Extract power in each frequency band using Welch's method.

        Args:
            sig: Preprocessed signal array

        Returns:
            BandPower dataclass with theta, alpha, beta, gamma values
        """
        if len(sig) < self.sampling_rate // 2:
            return BandPower(theta=0.0, alpha=0.0, beta=0.0, gamma=0.0)

        if BRAINFLOW_FILTER_AVAILABLE:
            return self._extract_band_powers_brainflow(sig)
        else:
            return self._extract_band_powers_scipy(sig)

    def _extract_band_powers_brainflow(self, sig: np.ndarray) -> BandPower:
        """Extract band powers using BrainFlow"""
        try:
            nfft = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            psd = DataFilter.get_psd_welch(
                sig, nfft, nfft // 2,
                self.sampling_rate, WindowOperations.HAMMING
            )

            powers = {}
            for band_name, (low_freq, high_freq) in self.BANDS.items():
                power = DataFilter.get_band_power(psd, low_freq, high_freq)
                powers[band_name] = float(power) if power > 0 else 0.0

            return BandPower(**powers)

        except Exception as e:
            logger.warning(f"BrainFlow band power warning: {e}")
            return BandPower(theta=0.0, alpha=0.0, beta=0.0, gamma=0.0)

    def _extract_band_powers_scipy(self, sig: np.ndarray) -> BandPower:
        """Extract band powers using SciPy (fallback)"""
        try:
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(
                sig,
                fs=self.sampling_rate,
                nperseg=min(256, len(sig)),
                noverlap=128
            )

            powers = {}
            for band_name, (low_freq, high_freq) in self.BANDS.items():
                # Find frequency indices
                idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                # Integrate power in band
                if np.any(idx):
                    power = np.trapz(psd[idx], freqs[idx])
                    powers[band_name] = float(power) if power > 0 else 0.0
                else:
                    powers[band_name] = 0.0

            return BandPower(**powers)

        except Exception as e:
            logger.warning(f"SciPy band power warning: {e}")
            return BandPower(theta=0.0, alpha=0.0, beta=0.0, gamma=0.0)

    def calculate_differential_entropy(self, signal: np.ndarray) -> float:
        """
        Calculate differential entropy (better for emotion recognition).

        DE = 0.5 * log(2 * pi * e * variance)

        Higher DE indicates more complex/variable signal.

        Args:
            signal: Preprocessed signal array

        Returns:
            Differential entropy value
        """
        variance = np.var(signal)
        if variance > 0:
            return float(0.5 * np.log(2 * np.pi * np.e * variance))
        return 0.0

    def process_window(
        self,
        data: np.ndarray,
        eeg_channels: List[int],
        channel_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Process a window of EEG data across all channels.

        Args:
            data: Raw data array from board (all channels)
            eeg_channels: List of EEG channel indices
            channel_names: Optional list of channel names (F3, F4, etc.)

        Returns:
            Dict with band_powers and differential_entropy per channel
        """
        results = {
            "band_powers": {},
            "differential_entropy": {}
        }

        if channel_names is None:
            channel_names = [f"ch_{i+1}" for i in range(len(eeg_channels))]

        for i, ch_idx in enumerate(eeg_channels):
            ch_name = channel_names[i] if i < len(channel_names) else f"ch_{i+1}"

            try:
                # Preprocess
                clean_signal = self.preprocess(data, ch_idx)

                # Extract features
                results["band_powers"][ch_name] = self.extract_band_powers(clean_signal)
                results["differential_entropy"][ch_name] = self.calculate_differential_entropy(clean_signal)

            except Exception as e:
                logger.warning(f"Channel {ch_name} processing warning: {e}")
                results["band_powers"][ch_name] = BandPower(0, 0, 0, 0)
                results["differential_entropy"][ch_name] = 0.0

        return results

    def normalize_powers(self, band_powers: Dict[str, BandPower]) -> Dict[str, BandPower]:
        """
        Normalize band powers across channels to 0-1 range.

        Useful for comparing across sessions or individuals.

        Args:
            band_powers: Dict of channel_name -> BandPower

        Returns:
            Normalized band powers
        """
        if not band_powers:
            return band_powers

        # Find max values across all channels
        max_theta = max(bp.theta for bp in band_powers.values()) or 1
        max_alpha = max(bp.alpha for bp in band_powers.values()) or 1
        max_beta = max(bp.beta for bp in band_powers.values()) or 1
        max_gamma = max(bp.gamma for bp in band_powers.values()) or 1

        normalized = {}
        for ch_name, bp in band_powers.items():
            normalized[ch_name] = BandPower(
                theta=bp.theta / max_theta,
                alpha=bp.alpha / max_alpha,
                beta=bp.beta / max_beta,
                gamma=bp.gamma / max_gamma
            )

        return normalized
