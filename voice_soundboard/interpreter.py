"""
Natural Language Style Interpreter.

Maps natural language style hints to synthesis parameters.
E.g., "warmly" -> speed=0.95, voice with warm style
"""

import re
from typing import Optional
from dataclasses import dataclass

from voice_soundboard.config import KOKORO_VOICES, VOICE_PRESETS


@dataclass
class StyleParams:
    """Interpreted style parameters."""
    voice: Optional[str] = None
    speed: Optional[float] = None
    preset: Optional[str] = None
    confidence: float = 0.0  # How confident we are in the interpretation


# Style keywords mapped to parameters
STYLE_KEYWORDS = {
    # Speed modifiers
    "quickly": {"speed": 1.2},
    "fast": {"speed": 1.2},
    "rapidly": {"speed": 1.3},
    "slowly": {"speed": 0.8},
    "slow": {"speed": 0.85},
    "carefully": {"speed": 0.9},
    "deliberately": {"speed": 0.85},

    # Energy/tone
    "excitedly": {"speed": 1.15, "style_prefer": ["bright", "energetic", "playful"]},
    "enthusiastically": {"speed": 1.1, "style_prefer": ["bright", "friendly"]},
    "calmly": {"speed": 0.9, "style_prefer": ["calm", "soft", "gentle"]},
    "gently": {"speed": 0.9, "style_prefer": ["soft", "gentle", "warm"]},
    "warmly": {"speed": 0.95, "style_prefer": ["warm", "friendly", "caring"]},
    "coldly": {"speed": 1.0, "style_prefer": ["neutral", "clear"]},
    "seriously": {"speed": 0.95, "style_prefer": ["authoritative", "deep"]},
    "playfully": {"speed": 1.05, "style_prefer": ["playful", "bright", "youthful"]},
    "mysteriously": {"speed": 0.85, "style_prefer": ["soft", "deep"]},
    "dramatically": {"speed": 0.9, "style_prefer": ["powerful", "authoritative"]},
    "cheerfully": {"speed": 1.1, "style_prefer": ["bright", "friendly", "jolly"]},
    "sadly": {"speed": 0.85, "style_prefer": ["soft", "gentle"]},
    "angrily": {"speed": 1.1, "style_prefer": ["powerful", "confident"]},
    "nervously": {"speed": 1.15, "style_prefer": ["soft", "youthful"]},
    "confidently": {"speed": 1.0, "style_prefer": ["confident", "professional", "authoritative"]},
    "softly": {"speed": 0.9, "style_prefer": ["soft", "gentle", "whisper"]},
    "loudly": {"speed": 1.05, "style_prefer": ["powerful", "bright"]},

    # Character styles
    "like a narrator": {"preset": "narrator"},
    "narratively": {"preset": "narrator"},
    "like an announcer": {"preset": "announcer"},
    "like a storyteller": {"preset": "storyteller"},
    "like whispering": {"preset": "whisper"},
    "whispered": {"preset": "whisper"},
    "professionally": {"speed": 1.0, "style_prefer": ["professional", "clear", "neutral"]},
    "casually": {"speed": 1.05, "style_prefer": ["friendly", "playful"]},

    # Gender preferences
    "in a male voice": {"gender_prefer": "male"},
    "in a female voice": {"gender_prefer": "female"},
    "masculine": {"gender_prefer": "male"},
    "feminine": {"gender_prefer": "female"},

    # Accent preferences
    "with a british accent": {"accent_prefer": "british"},
    "british": {"accent_prefer": "british"},
    "american": {"accent_prefer": "american"},
    "with an american accent": {"accent_prefer": "american"},
}


def find_best_voice(
    style_prefer: list[str] = None,
    gender_prefer: str = None,
    accent_prefer: str = None,
) -> Optional[str]:
    """Find the best matching voice based on preferences."""

    candidates = []

    for voice_id, info in KOKORO_VOICES.items():
        score = 0

        # Gender match
        if gender_prefer and info.get("gender") == gender_prefer:
            score += 10

        # Accent match
        if accent_prefer and info.get("accent") == accent_prefer:
            score += 5

        # Style match
        if style_prefer:
            voice_style = info.get("style", "").lower()
            for pref in style_prefer:
                if pref.lower() in voice_style:
                    score += 3

        if score > 0:
            candidates.append((voice_id, score))

    if candidates:
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None


def interpret_style(style_hint: str) -> StyleParams:
    """
    Interpret a natural language style hint into synthesis parameters.

    Args:
        style_hint: Natural language like "warmly", "quickly and excitedly"

    Returns:
        StyleParams with interpreted voice, speed, etc.

    Examples:
        >>> interpret_style("warmly")
        StyleParams(speed=0.95, voice="af_bella", ...)

        >>> interpret_style("like a narrator, slowly")
        StyleParams(preset="narrator", speed=0.8, ...)
    """
    if not style_hint:
        return StyleParams(confidence=0.0)

    hint_lower = style_hint.lower().strip()

    # Accumulated parameters
    speed_values = []
    style_prefs = []
    gender_pref = None
    accent_pref = None
    preset = None
    matched_keywords = 0

    # Check each keyword
    for keyword, params in STYLE_KEYWORDS.items():
        if keyword in hint_lower:
            matched_keywords += 1

            if "speed" in params:
                speed_values.append(params["speed"])
            if "style_prefer" in params:
                style_prefs.extend(params["style_prefer"])
            if "gender_prefer" in params:
                gender_pref = params["gender_prefer"]
            if "accent_prefer" in params:
                accent_pref = params["accent_prefer"]
            if "preset" in params:
                preset = params["preset"]

    # Calculate final speed (average if multiple)
    final_speed = None
    if speed_values:
        final_speed = sum(speed_values) / len(speed_values)

    # Find best voice if we have preferences
    voice = None
    if style_prefs or gender_pref or accent_pref:
        voice = find_best_voice(style_prefs, gender_pref, accent_pref)

    # Calculate confidence based on matches
    confidence = min(1.0, matched_keywords * 0.3) if matched_keywords > 0 else 0.0

    return StyleParams(
        voice=voice,
        speed=final_speed,
        preset=preset,
        confidence=confidence,
    )


def apply_style_to_params(
    style_hint: str,
    voice: Optional[str] = None,
    speed: Optional[float] = None,
    preset: Optional[str] = None,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Apply style interpretation to existing parameters.

    Explicit parameters override interpreted ones.

    Returns:
        (voice, speed, preset) tuple with style applied
    """
    interpreted = interpret_style(style_hint)

    # Explicit params take precedence
    final_voice = voice or interpreted.voice
    final_speed = speed if speed is not None else interpreted.speed
    final_preset = preset or interpreted.preset

    return final_voice, final_speed, final_preset


if __name__ == "__main__":
    # Quick test
    test_hints = [
        "warmly",
        "quickly and excitedly",
        "like a narrator, slowly",
        "in a british accent, professionally",
        "whispered mysteriously",
        "cheerfully in a male voice",
    ]

    print("Style Interpretation Tests:")
    print("-" * 50)

    for hint in test_hints:
        result = interpret_style(hint)
        print(f"\n'{hint}':")
        print(f"  voice: {result.voice}")
        print(f"  speed: {result.speed}")
        print(f"  preset: {result.preset}")
        print(f"  confidence: {result.confidence:.1%}")
