"""
Voice Studio AI Assistant

Provides natural language parsing for voice design and
AI-powered suggestions for preset creation.
"""

import re
import logging
from typing import Optional, Any
from dataclasses import dataclass

from voice_soundboard.presets.schema import AcousticParams, VoicePreset

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A parameter adjustment suggestion."""
    param: str
    value: float
    reason: str
    confidence: float = 1.0


class StudioAIAssistant:
    """
    AI-powered suggestions for voice design.

    Parses natural language descriptions and maps them to
    acoustic parameter adjustments.

    Example:
        >>> assistant = StudioAIAssistant()
        >>> suggestions = assistant.parse_description("make it deeper and more authoritative")
        >>> # Returns: {"formant_ratio": 0.88, "jitter_percent": 1.2, ...}
    """

    # Keyword mappings for parameter inference
    # Format: keyword -> dict of parameter adjustments
    VOICE_QUALITIES = {
        # Depth/Brightness (formant_ratio)
        "deep": {"formant_ratio": 0.90},
        "deeper": {"formant_ratio": 0.87},
        "very deep": {"formant_ratio": 0.85},
        "bass": {"formant_ratio": 0.85},
        "resonant": {"formant_ratio": 0.88},
        "booming": {"formant_ratio": 0.85, "jitter_percent": 0.8},

        "bright": {"formant_ratio": 1.08},
        "brighter": {"formant_ratio": 1.10},
        "very bright": {"formant_ratio": 1.12},
        "light": {"formant_ratio": 1.10},
        "high": {"formant_ratio": 1.08},
        "thin": {"formant_ratio": 1.12, "breath_intensity": 0.08},

        # Character voices
        "warm": {"formant_ratio": 0.96, "breath_intensity": 0.18},
        "warmer": {"formant_ratio": 0.94, "breath_intensity": 0.20},
        "cozy": {"formant_ratio": 0.95, "breath_intensity": 0.20},
        "inviting": {"formant_ratio": 0.96, "breath_intensity": 0.15},

        "husky": {"formant_ratio": 0.93, "jitter_percent": 1.5, "shimmer_percent": 3.5},
        "gravelly": {"formant_ratio": 0.90, "jitter_percent": 2.0, "shimmer_percent": 4.0},
        "raspy": {"jitter_percent": 2.5, "shimmer_percent": 5.0},
        "rough": {"jitter_percent": 2.0, "shimmer_percent": 4.0},
        "gritty": {"jitter_percent": 2.2, "shimmer_percent": 4.5},

        "smooth": {"jitter_percent": 0.3, "shimmer_percent": 1.5},
        "silky": {"jitter_percent": 0.2, "shimmer_percent": 1.0, "formant_ratio": 1.02},
        "velvety": {"jitter_percent": 0.3, "shimmer_percent": 1.5, "breath_intensity": 0.12},

        # Age-related
        "youthful": {"formant_ratio": 1.08, "pitch_drift_cents": 12.0},
        "young": {"formant_ratio": 1.06, "pitch_drift_cents": 10.0},
        "childlike": {"formant_ratio": 1.15, "pitch_drift_cents": 15.0, "timing_variation_ms": 18.0},
        "teenage": {"formant_ratio": 1.05, "pitch_drift_cents": 12.0},

        "aged": {"formant_ratio": 0.94, "jitter_percent": 1.8, "shimmer_percent": 3.5},
        "elderly": {"formant_ratio": 0.92, "jitter_percent": 2.2, "shimmer_percent": 4.0, "breath_intensity": 0.25},
        "mature": {"formant_ratio": 0.96, "jitter_percent": 1.0},
        "seasoned": {"formant_ratio": 0.95, "jitter_percent": 1.2, "pitch_drift_cents": 10.0},
        "wise": {"formant_ratio": 0.94, "speed_factor": 0.92, "timing_variation_ms": 15.0},

        # Breathiness
        "breathy": {"breath_intensity": 0.35, "breath_volume_db": -22.0, "shimmer_percent": 4.0},
        "airy": {"breath_intensity": 0.30, "shimmer_percent": 3.5},
        "intimate": {"breath_intensity": 0.28, "breath_volume_db": -20.0, "formant_ratio": 0.97},
        "whispered": {"breath_intensity": 0.40, "shimmer_percent": 6.0},
        "soft": {"breath_intensity": 0.22, "speed_factor": 0.95},

        # Clarity
        "clear": {"breath_intensity": 0.05, "jitter_percent": 0.3, "shimmer_percent": 1.0},
        "crisp": {"breath_intensity": 0.03, "jitter_percent": 0.2},
        "articulate": {"timing_variation_ms": 5.0, "jitter_percent": 0.3},
        "precise": {"timing_variation_ms": 3.0, "jitter_percent": 0.2},

        # Naturalness
        "natural": {"jitter_percent": 0.8, "shimmer_percent": 2.5, "breath_intensity": 0.18},
        "human": {"jitter_percent": 1.0, "shimmer_percent": 2.8, "breath_intensity": 0.20, "pitch_drift_cents": 10.0},
        "organic": {"jitter_percent": 0.9, "shimmer_percent": 2.5, "timing_variation_ms": 12.0},
        "lifelike": {"jitter_percent": 0.8, "shimmer_percent": 2.5, "breath_intensity": 0.15},

        "robotic": {"jitter_percent": 0.0, "shimmer_percent": 0.0, "breath_intensity": 0.0, "pitch_drift_cents": 0.0},
        "mechanical": {"jitter_percent": 0.1, "shimmer_percent": 0.5, "breath_intensity": 0.0, "timing_variation_ms": 2.0},
        "synthetic": {"jitter_percent": 0.2, "shimmer_percent": 0.8, "breath_intensity": 0.05},

        # Speed/Energy
        "fast": {"speed_factor": 1.25},
        "faster": {"speed_factor": 1.35},
        "quick": {"speed_factor": 1.20, "timing_variation_ms": 8.0},
        "rapid": {"speed_factor": 1.40},
        "slow": {"speed_factor": 0.85},
        "slower": {"speed_factor": 0.80},
        "measured": {"speed_factor": 0.90, "timing_variation_ms": 8.0},
        "deliberate": {"speed_factor": 0.88, "timing_variation_ms": 6.0},

        # Energy level
        "energetic": {"pitch_drift_cents": 15.0, "timing_variation_ms": 8.0, "speed_factor": 1.10},
        "dynamic": {"pitch_drift_cents": 14.0, "timing_variation_ms": 10.0},
        "lively": {"pitch_drift_cents": 12.0, "speed_factor": 1.08},
        "animated": {"pitch_drift_cents": 14.0, "timing_variation_ms": 12.0},

        "calm": {"pitch_drift_cents": 5.0, "timing_variation_ms": 15.0, "speed_factor": 0.95},
        "relaxed": {"pitch_drift_cents": 6.0, "timing_variation_ms": 18.0, "speed_factor": 0.92},
        "soothing": {"pitch_drift_cents": 4.0, "breath_intensity": 0.20, "speed_factor": 0.90},
        "peaceful": {"pitch_drift_cents": 3.0, "timing_variation_ms": 15.0, "speed_factor": 0.88},
        "serene": {"pitch_drift_cents": 3.0, "timing_variation_ms": 12.0, "breath_intensity": 0.18},

        # Professional qualities
        "authoritative": {"formant_ratio": 0.88, "jitter_percent": 1.0, "timing_variation_ms": 6.0},
        "commanding": {"formant_ratio": 0.86, "jitter_percent": 0.8, "speed_factor": 0.95},
        "confident": {"formant_ratio": 0.95, "timing_variation_ms": 5.0, "pitch_drift_cents": 6.0},
        "professional": {"jitter_percent": 0.5, "timing_variation_ms": 6.0, "breath_intensity": 0.10},
        "broadcast": {"formant_ratio": 0.98, "jitter_percent": 0.4, "timing_variation_ms": 4.0},

        # Narrative styles
        "narrator": {"formant_ratio": 0.96, "breath_intensity": 0.15, "pitch_drift_cents": 8.0},
        "storyteller": {"pitch_drift_cents": 12.0, "timing_variation_ms": 15.0, "breath_intensity": 0.18},
        "documentary": {"formant_ratio": 0.95, "breath_intensity": 0.12, "timing_variation_ms": 8.0},
        "audiobook": {"breath_intensity": 0.15, "pitch_drift_cents": 10.0, "timing_variation_ms": 12.0},
    }

    # Modifiers that adjust intensity
    MODIFIERS = {
        "very": 1.5,
        "extremely": 2.0,
        "slightly": 0.5,
        "a bit": 0.5,
        "somewhat": 0.7,
        "much": 1.3,
        "more": 1.2,
        "less": 0.7,
        "little": 0.6,
    }

    def __init__(self):
        self._catalog = None

    def _get_catalog(self):
        """Lazy-load preset catalog."""
        if self._catalog is None:
            try:
                from voice_soundboard.presets import get_catalog
                self._catalog = get_catalog()
            except ImportError:
                logger.warning("Preset catalog not available")
        return self._catalog

    def parse_description(
        self,
        description: str,
        base_params: Optional[AcousticParams] = None,
    ) -> dict[str, Any]:
        """
        Parse natural language description into acoustic parameters.

        Args:
            description: Natural language voice description
            base_params: Optional base parameters to modify

        Returns:
            Dictionary of parameter adjustments with metadata

        Example:
            >>> parse_description("make it deeper and more authoritative")
            {
                "params": {"formant_ratio": 0.88, "jitter_percent": 1.0},
                "suggestions": [...],
                "explanation": "Applied: deep, authoritative"
            }
        """
        description_lower = description.lower()
        suggestions = []
        accumulated_params = {}
        matched_keywords = []

        # Find modifier for intensity scaling
        modifier_scale = 1.0
        for mod, scale in self.MODIFIERS.items():
            if mod in description_lower:
                modifier_scale = scale
                break

        # Match keywords
        for keyword, params in self.VOICE_QUALITIES.items():
            if keyword in description_lower:
                matched_keywords.append(keyword)

                for param, value in params.items():
                    # Get current or default value
                    if base_params:
                        current = getattr(base_params, param, self._get_default(param))
                    else:
                        current = self._get_default(param)

                    # Calculate adjusted value with modifier
                    if param in ["formant_ratio", "speed_factor"]:
                        # For ratio params, scale the deviation from 1.0
                        deviation = value - 1.0
                        adjusted = 1.0 + (deviation * modifier_scale)
                    else:
                        # For other params, scale toward the target
                        adjusted = current + (value - current) * modifier_scale

                    # Clamp to valid range
                    adjusted = self._clamp_param(param, adjusted)

                    # Accumulate (average if multiple keywords set same param)
                    if param in accumulated_params:
                        accumulated_params[param] = (accumulated_params[param] + adjusted) / 2
                    else:
                        accumulated_params[param] = adjusted

                    suggestions.append(Suggestion(
                        param=param,
                        value=adjusted,
                        reason=f"'{keyword}' suggests {param}={adjusted:.2f}",
                        confidence=0.9,
                    ))

        # Generate explanation
        if matched_keywords:
            explanation = f"Applied voice qualities: {', '.join(matched_keywords)}"
        else:
            explanation = "No specific voice qualities matched. Try keywords like: deep, warm, breathy, energetic, calm"

        return {
            "params": accumulated_params,
            "suggestions": [
                {"param": s.param, "value": s.value, "reason": s.reason}
                for s in suggestions
            ],
            "matched_keywords": matched_keywords,
            "explanation": explanation,
            "modifier": modifier_scale if modifier_scale != 1.0 else None,
        }

    def _get_default(self, param: str) -> float:
        """Get default value for a parameter."""
        defaults = {
            "formant_ratio": 1.0,
            "breath_intensity": 0.15,
            "breath_volume_db": -28.0,
            "jitter_percent": 0.5,
            "shimmer_percent": 2.0,
            "pitch_drift_cents": 8.0,
            "timing_variation_ms": 10.0,
            "speed_factor": 1.0,
            "pitch_shift_semitones": 0.0,
        }
        return defaults.get(param, 0.0)

    def _clamp_param(self, param: str, value: float) -> float:
        """Clamp parameter to valid range."""
        ranges = {
            "formant_ratio": (0.8, 1.2),
            "breath_intensity": (0.0, 0.5),
            "breath_volume_db": (-40.0, -10.0),
            "jitter_percent": (0.0, 3.0),
            "shimmer_percent": (0.0, 10.0),
            "pitch_drift_cents": (0.0, 20.0),
            "timing_variation_ms": (0.0, 30.0),
            "speed_factor": (0.5, 2.0),
            "pitch_shift_semitones": (-12.0, 12.0),
        }
        if param in ranges:
            min_val, max_val = ranges[param]
            return max(min_val, min(max_val, value))
        return value

    def suggest_presets(
        self,
        description: str,
        limit: int = 3,
    ) -> list[VoicePreset]:
        """
        Find existing presets matching a description.

        Uses the preset catalog's search functionality.

        Args:
            description: Natural language description
            limit: Maximum number of presets to return

        Returns:
            List of matching VoicePreset objects
        """
        catalog = self._get_catalog()
        if catalog is None:
            return []

        results = catalog.search(description, limit=limit)
        return [r.preset for r in results]

    def generate_voice_prompt(self, params: AcousticParams) -> str:
        """
        Generate a Qwen3-style voice prompt from acoustic parameters.

        Creates a natural language description suitable for
        voice prompt-based TTS systems.

        Args:
            params: AcousticParams to describe

        Returns:
            Natural language voice prompt string
        """
        parts = []

        # Describe formant/depth
        if params.formant_ratio < 0.90:
            parts.append("very deep, resonant")
        elif params.formant_ratio < 0.95:
            parts.append("deep")
        elif params.formant_ratio > 1.10:
            parts.append("very bright, light")
        elif params.formant_ratio > 1.05:
            parts.append("bright")

        # Describe breathiness
        if params.breath_intensity > 0.30:
            parts.append("breathy, intimate")
        elif params.breath_intensity > 0.20:
            parts.append("with natural breaths")
        elif params.breath_intensity < 0.05:
            parts.append("clear, without breaths")

        # Describe texture
        if params.jitter_percent > 1.5:
            parts.append("gravelly, textured")
        elif params.jitter_percent > 0.8:
            parts.append("naturally textured")
        elif params.jitter_percent < 0.3:
            parts.append("smooth")

        # Describe speed
        if params.speed_factor < 0.85:
            parts.append("slow and measured")
        elif params.speed_factor > 1.20:
            parts.append("quick and energetic")

        # Describe pitch variation
        if params.pitch_drift_cents > 12:
            parts.append("dynamic intonation")
        elif params.pitch_drift_cents < 4:
            parts.append("steady pitch")

        if not parts:
            return "Natural, balanced voice with moderate variation"

        return ", ".join(parts).capitalize()

    def explain_params(self, params: AcousticParams) -> dict:
        """
        Generate human-readable explanations for each parameter.

        Useful for UI tooltips and educational display.
        """
        explanations = {
            "formant_ratio": {
                "name": "Voice Depth",
                "value": params.formant_ratio,
                "range": "0.8 (deep) to 1.2 (bright)",
                "description": self._describe_formant(params.formant_ratio),
            },
            "breath_intensity": {
                "name": "Breathiness",
                "value": params.breath_intensity,
                "range": "0.0 (none) to 0.5 (very breathy)",
                "description": self._describe_breath(params.breath_intensity),
            },
            "jitter_percent": {
                "name": "Voice Texture",
                "value": params.jitter_percent,
                "range": "0% (smooth) to 3% (gravelly)",
                "description": self._describe_jitter(params.jitter_percent),
            },
            "pitch_drift_cents": {
                "name": "Pitch Variation",
                "value": params.pitch_drift_cents,
                "range": "0 (monotone) to 20 cents (varied)",
                "description": self._describe_drift(params.pitch_drift_cents),
            },
            "timing_variation_ms": {
                "name": "Timing Feel",
                "value": params.timing_variation_ms,
                "range": "0 ms (precise) to 30 ms (relaxed)",
                "description": self._describe_timing(params.timing_variation_ms),
            },
            "speed_factor": {
                "name": "Speaking Speed",
                "value": params.speed_factor,
                "range": "0.5x (slow) to 2.0x (fast)",
                "description": self._describe_speed(params.speed_factor),
            },
        }
        return explanations

    def _describe_formant(self, value: float) -> str:
        if value < 0.88:
            return "Very deep, large-sounding voice"
        elif value < 0.95:
            return "Deeper than average, warm tone"
        elif value > 1.10:
            return "Very bright, smaller-sounding voice"
        elif value > 1.02:
            return "Brighter than average, forward tone"
        return "Neutral, balanced depth"

    def _describe_breath(self, value: float) -> str:
        if value > 0.30:
            return "Very breathy, intimate quality"
        elif value > 0.18:
            return "Natural breath sounds between phrases"
        elif value < 0.05:
            return "No audible breaths, clean sound"
        return "Subtle breath presence"

    def _describe_jitter(self, value: float) -> str:
        if value > 1.5:
            return "Gravelly, rough texture"
        elif value > 0.8:
            return "Natural voice texture"
        elif value < 0.3:
            return "Smooth, polished sound"
        return "Moderate natural variation"

    def _describe_drift(self, value: float) -> str:
        if value > 12:
            return "Expressive, dynamic pitch movement"
        elif value < 4:
            return "Steady, controlled pitch"
        return "Natural pitch variation"

    def _describe_timing(self, value: float) -> str:
        if value > 18:
            return "Relaxed, conversational timing"
        elif value < 5:
            return "Precise, broadcast-style timing"
        return "Natural timing variation"

    def _describe_speed(self, value: float) -> str:
        if value < 0.85:
            return "Slow, deliberate pace"
        elif value > 1.20:
            return "Fast, energetic delivery"
        return "Normal speaking pace"

    def get_keywords(self) -> list[str]:
        """Get all supported voice quality keywords."""
        return sorted(self.VOICE_QUALITIES.keys())

    def get_qualities_list(self) -> list[str]:
        """Alias for get_keywords() for UI convenience."""
        return self.get_keywords()

    def suggest_from_params(self, params: AcousticParams) -> str:
        """Alias for generate_voice_prompt() for UI convenience."""
        return self.generate_voice_prompt(params)

    def get_keyword_info(self, keyword: str) -> Optional[dict]:
        """Get info about a specific keyword."""
        if keyword in self.VOICE_QUALITIES:
            params = self.VOICE_QUALITIES[keyword]
            return {
                "keyword": keyword,
                "params": params,
                "description": f"Adjusts: {', '.join(f'{k}={v}' for k, v in params.items())}",
            }
        return None
