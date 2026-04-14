# reusable_lib/eeg/experience.py
"""
Experience Format Generation for Neural Resonance.

Translates EEG brain activity into AI-readable "felt experience" format.
This is the core innovation - enabling AI agents to perceive how humans
experience music, content, or any stimulus.

Based on:
- Frontal asymmetry (F4-F3 alpha) for valence (positive/negative emotion)
- Beta/alpha ratio for arousal (calm/excited)
- Theta-beta coupling for emotional engagement
- Gamma bursts for "chills" / peak experiences

Usage:
    from reusable_lib.eeg import EmotionMapper, ListeningSession

    mapper = EmotionMapper()

    # Process a moment of EEG data
    moment = mapper.process_moment(band_powers, timestamp_ms=1000)
    print(f"Valence: {moment.valence}, Arousal: {moment.arousal}")

    # Create and save a listening session
    session = ListeningSession(
        session_id="listen_001",
        track_id="track_123",
        track_title="My Song",
        listener="User",
        duration_ms=180000,
        moments=[moment1, moment2, ...]
    )
    session.save_to_file("./session.json")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import json
import os
from datetime import datetime
import logging

from .processor import BandPower

logger = logging.getLogger(__name__)


@dataclass
class MomentExperience:
    """A single moment of felt experience during content consumption"""
    timestamp_ms: int
    track_position: str  # e.g., "1:47 - bridge entry"

    # Derived emotional dimensions
    valence: float        # -1 (negative) to +1 (positive)
    arousal: float        # 0 (calm) to 1 (excited)
    attention: float      # 0 (distracted) to 1 (focused)
    engagement: float     # 0 (passive) to 1 (immersed)

    # Event flags
    attention_shift: bool = False
    emotional_peak: bool = False
    possible_chills: bool = False

    # Raw data (optional, for detailed analysis)
    channels: Optional[Dict[str, Dict[str, float]]] = None
    musical_context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling optional fields"""
        result = {
            "timestamp_ms": self.timestamp_ms,
            "track_position": self.track_position,
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "attention": round(self.attention, 3),
            "engagement": round(self.engagement, 3),
            "attention_shift": self.attention_shift,
            "emotional_peak": self.emotional_peak,
            "possible_chills": self.possible_chills,
        }
        if self.channels:
            result["channels"] = self.channels
        if self.musical_context:
            result["musical_context"] = self.musical_context
        return result


class EmotionMapper:
    """Maps EEG features to emotional dimensions"""

    def __init__(self, baseline: Optional[Dict] = None):
        """
        Initialize the emotion mapper.

        Args:
            baseline: Optional baseline measurements for personalization
        """
        self.baseline = baseline or {}
        self.previous_attention = 0.5
        self.previous_arousal = 0.5

    def calculate_valence(self, band_powers: Dict[str, BandPower]) -> float:
        """
        Calculate emotional valence from frontal asymmetry.

        Based on research:
        - Greater right frontal alpha (F4 > F3) = positive/approach emotion
        - Greater left frontal alpha (F3 > F4) = negative/withdrawal emotion
        - Frontal theta also contributes to emotional engagement

        Args:
            band_powers: Dict of channel_name -> BandPower

        Returns:
            Valence score from -1 (negative) to +1 (positive)
        """
        # Try to get F3 and F4 (frontal left/right)
        f3 = band_powers.get("F3") or band_powers.get("ch_3")
        f4 = band_powers.get("F4") or band_powers.get("ch_4")

        if not f3 or not f4:
            # Fallback: use first two channels
            channels = list(band_powers.values())
            if len(channels) >= 2:
                f3, f4 = channels[0], channels[1]
            else:
                return 0.0

        f3_alpha = f3.alpha if hasattr(f3, 'alpha') else 0.5
        f4_alpha = f4.alpha if hasattr(f4, 'alpha') else 0.5
        f3_theta = f3.theta if hasattr(f3, 'theta') else 0.5
        f4_theta = f4.theta if hasattr(f4, 'theta') else 0.5

        # Frontal alpha asymmetry
        total_alpha = f4_alpha + f3_alpha
        if total_alpha > 0.001:
            asymmetry = (f4_alpha - f3_alpha) / total_alpha
        else:
            asymmetry = 0.0

        # Average frontal theta (emotional engagement)
        avg_theta = (f3_theta + f4_theta) / 2

        # Combine: asymmetry weighted by theta engagement
        valence = np.tanh(asymmetry * 2 + avg_theta * 0.3)

        return float(np.clip(valence, -1, 1))

    def calculate_arousal(self, band_powers: Dict[str, BandPower]) -> float:
        """
        Calculate arousal level from beta/alpha ratio.

        Higher beta relative to alpha = more aroused/excited state.

        Args:
            band_powers: Dict of channel_name -> BandPower

        Returns:
            Arousal score from 0 (calm) to 1 (excited)
        """
        total_beta = 0.0
        total_alpha = 0.0
        count = 0

        for powers in band_powers.values():
            if hasattr(powers, 'beta') and hasattr(powers, 'alpha'):
                total_beta += powers.beta
                total_alpha += powers.alpha
                count += 1

        if count > 0 and total_alpha > 0.001:
            ratio = total_beta / total_alpha
            # Normalize: typical ratio is 0.5-3, map to 0-1
            arousal = np.clip(ratio / 3, 0, 1)
            return float(arousal)

        return 0.5

    def calculate_attention(self, band_powers: Dict[str, BandPower]) -> float:
        """
        Calculate attention level from theta/beta ratio and gamma.

        Lower theta/beta ratio = more focused attention.
        Higher gamma = active cognitive processing.

        Args:
            band_powers: Dict of channel_name -> BandPower

        Returns:
            Attention score from 0 (distracted) to 1 (focused)
        """
        # Get frontal channels for theta/beta ratio
        f3 = band_powers.get("F3") or band_powers.get("ch_3")
        if f3 and hasattr(f3, 'theta') and hasattr(f3, 'beta') and f3.beta > 0.001:
            tbr = f3.theta / f3.beta
            # Lower TBR = higher attention
            attention_from_tbr = 1 - np.clip(tbr / 2, 0, 1)
        else:
            attention_from_tbr = 0.5

        # Average gamma across channels (indicates focused processing)
        gamma_values = [
            bp.gamma for bp in band_powers.values()
            if hasattr(bp, 'gamma')
        ]
        avg_gamma = np.mean(gamma_values) if gamma_values else 0.5

        # Normalize gamma contribution
        gamma_attention = np.clip(avg_gamma * 2, 0, 1)

        # Combine
        attention = (attention_from_tbr + gamma_attention) / 2

        return float(np.clip(attention, 0, 1))

    def calculate_engagement(self, arousal: float, attention: float) -> float:
        """
        Calculate overall engagement from arousal and attention.

        Engagement = how "into it" the listener is.

        Args:
            arousal: Calculated arousal score
            attention: Calculated attention score

        Returns:
            Engagement score from 0 (passive) to 1 (immersed)
        """
        # Geometric mean gives better "both need to be high" behavior
        return float(np.sqrt(arousal * attention))

    def detect_chills(
        self,
        band_powers: Dict[str, BandPower],
        arousal: float,
        attention: float
    ) -> bool:
        """
        Detect possible musical chills/frisson.

        Musical chills are characterized by:
        - Sudden gamma burst (cognitive integration)
        - High frontal theta (emotional processing)
        - High arousal spike
        - High attention/engagement

        Args:
            band_powers: Dict of channel_name -> BandPower
            arousal: Current arousal level
            attention: Current attention level

        Returns:
            True if chills indicators are present
        """
        # Average gamma across all channels
        gamma_values = [
            bp.gamma for bp in band_powers.values()
            if hasattr(bp, 'gamma')
        ]
        avg_gamma = np.mean(gamma_values) if gamma_values else 0

        # Frontal theta
        f3 = band_powers.get("F3") or band_powers.get("ch_3")
        f4 = band_powers.get("F4") or band_powers.get("ch_4")
        theta_values = []
        if f3 and hasattr(f3, 'theta'):
            theta_values.append(f3.theta)
        if f4 and hasattr(f4, 'theta'):
            theta_values.append(f4.theta)
        avg_theta = np.mean(theta_values) if theta_values else 0

        # Chills: high gamma + high theta + high arousal + high attention
        # Thresholds are empirical - may need calibration per individual
        if avg_gamma > 0.6 and avg_theta > 0.5 and arousal > 0.75 and attention > 0.75:
            return True

        return False

    def detect_attention_shift(self, current_attention: float) -> bool:
        """
        Detect sudden attention shifts.

        Args:
            current_attention: Current attention level

        Returns:
            True if significant attention shift detected
        """
        shift = abs(current_attention - self.previous_attention) > 0.2
        self.previous_attention = current_attention
        return shift

    def detect_emotional_peak(self, valence: float, arousal: float) -> bool:
        """
        Detect emotional peaks (strong positive emotion + high arousal).

        Args:
            valence: Current valence score
            arousal: Current arousal score

        Returns:
            True if emotional peak detected
        """
        return valence > 0.6 and arousal > 0.7

    def process_moment(
        self,
        band_powers: Dict[str, BandPower],
        timestamp_ms: int,
        track_position: str = "",
        musical_context: str = "",
        include_raw: bool = True
    ) -> MomentExperience:
        """
        Process a moment of EEG data into a MomentExperience.

        Args:
            band_powers: Dict of channel_name -> BandPower
            timestamp_ms: Timestamp in milliseconds
            track_position: Human-readable position (e.g., "1:47")
            musical_context: Description of what's happening musically
            include_raw: Whether to include raw channel data

        Returns:
            MomentExperience dataclass
        """
        # Calculate emotional dimensions
        valence = self.calculate_valence(band_powers)
        arousal = self.calculate_arousal(band_powers)
        attention = self.calculate_attention(band_powers)
        engagement = self.calculate_engagement(arousal, attention)

        # Detect events
        attention_shift = self.detect_attention_shift(attention)
        emotional_peak = self.detect_emotional_peak(valence, arousal)
        possible_chills = self.detect_chills(band_powers, arousal, attention)

        # Convert band powers to serializable format
        channels_dict = None
        if include_raw:
            channels_dict = {}
            for ch_name, powers in band_powers.items():
                if hasattr(powers, 'to_dict'):
                    channels_dict[ch_name] = powers.to_dict()
                elif hasattr(powers, 'theta'):
                    channels_dict[ch_name] = {
                        "theta": round(powers.theta, 4),
                        "alpha": round(powers.alpha, 4),
                        "beta": round(powers.beta, 4),
                        "gamma": round(powers.gamma, 4)
                    }

        return MomentExperience(
            timestamp_ms=timestamp_ms,
            track_position=track_position,
            valence=valence,
            arousal=arousal,
            attention=attention,
            engagement=engagement,
            attention_shift=attention_shift,
            emotional_peak=emotional_peak,
            possible_chills=possible_chills,
            channels=channels_dict,
            musical_context=musical_context
        )


@dataclass
class ListeningSession:
    """Complete listening session for a track or content piece"""
    session_id: str
    track_id: str
    track_title: str
    listener: str
    duration_ms: int
    moments: List[MomentExperience] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "track_id": self.track_id,
            "track_title": self.track_title,
            "listener": self.listener,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "summary": self.generate_summary(),
            "moments": [m.to_dict() for m in self.moments],
            "experience_narrative": self.generate_narrative()
        }

    def generate_summary(self) -> Dict[str, Any]:
        """Generate session summary statistics"""
        if not self.moments:
            return {
                "overall_valence": 0.0,
                "overall_arousal": 0.0,
                "peak_moments": [],
                "chills_count": 0
            }

        valences = [m.valence for m in self.moments]
        arousals = [m.arousal for m in self.moments]

        peak_moments = [
            m.timestamp_ms for m in self.moments
            if m.emotional_peak or m.possible_chills
        ]

        return {
            "overall_valence": round(float(np.mean(valences)), 3),
            "overall_arousal": round(float(np.mean(arousals)), 3),
            "valence_range": [round(float(min(valences)), 3), round(float(max(valences)), 3)],
            "arousal_range": [round(float(min(arousals)), 3), round(float(max(arousals)), 3)],
            "peak_moments": peak_moments[:10],  # Limit to 10
            "chills_count": sum(1 for m in self.moments if m.possible_chills),
            "attention_shifts": sum(1 for m in self.moments if m.attention_shift),
            "emotional_peaks": sum(1 for m in self.moments if m.emotional_peak)
        }

    def generate_narrative(self) -> str:
        """Generate natural language description for AI consumption"""
        if not self.moments:
            return "No listening data recorded."

        summary = self.generate_summary()
        valence = summary["overall_valence"]
        arousal = summary["overall_arousal"]

        # Determine emotional character
        if valence > 0.4 and arousal > 0.6:
            character = "joyful and energizing"
        elif valence > 0.4 and arousal > 0.3:
            character = "peaceful and pleasant"
        elif valence > 0.4:
            character = "calm and soothing"
        elif valence < -0.2 and arousal > 0.6:
            character = "intense and stirring"
        elif valence < -0.2:
            character = "melancholic and reflective"
        else:
            character = "contemplative and balanced"

        # Build narrative
        narrative = f"{self.listener} listened to '{self.track_title}'. "
        narrative += f"The experience was {character} "
        narrative += f"(valence: {valence:.2f}, arousal: {arousal:.2f}). "

        if summary["chills_count"] > 0:
            narrative += f"Musical chills detected {summary['chills_count']} time(s). "

        if summary["peak_moments"]:
            peak_times = [f"{t//1000//60}:{(t//1000)%60:02d}" for t in summary["peak_moments"][:3]]
            narrative += f"Emotional peaks at: {', '.join(peak_times)}. "

        if summary["attention_shifts"] > 3:
            narrative += f"The music held attention with {summary['attention_shifts']} notable moments. "

        return narrative.strip()

    def save_to_file(self, filepath: str) -> bool:
        """Save session to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Session saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['ListeningSession']:
        """Load session from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Reconstruct moments
            moments = []
            for m_data in data.get("moments", []):
                moments.append(MomentExperience(
                    timestamp_ms=m_data["timestamp_ms"],
                    track_position=m_data.get("track_position", ""),
                    valence=m_data["valence"],
                    arousal=m_data["arousal"],
                    attention=m_data["attention"],
                    engagement=m_data["engagement"],
                    attention_shift=m_data.get("attention_shift", False),
                    emotional_peak=m_data.get("emotional_peak", False),
                    possible_chills=m_data.get("possible_chills", False),
                    channels=m_data.get("channels"),
                    musical_context=m_data.get("musical_context", "")
                ))

            return cls(
                session_id=data["session_id"],
                track_id=data["track_id"],
                track_title=data["track_title"],
                listener=data["listener"],
                duration_ms=data["duration_ms"],
                moments=moments,
                created_at=data.get("created_at", datetime.now().isoformat())
            )
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
