"""
Voice Auto-Assignment for Multi-Speaker Dialogue.

Automatically assigns appropriate voices to speakers based on:
- Character name hints (gender, age)
- Explicit voice preferences
- Voice diversity (avoiding duplicate voices)
- Role-based defaults (narrator gets authoritative voice, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from voice_soundboard.config import KOKORO_VOICES
from voice_soundboard.dialogue.parser import Speaker, ParsedScript


class VoiceGender(Enum):
    """Voice gender categories."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(Enum):
    """Voice age categories."""
    YOUNG = "young"
    ADULT = "adult"
    ELDERLY = "elderly"


class VoiceStyle(Enum):
    """Voice style categories."""
    NEUTRAL = "neutral"
    WARM = "warm"
    AUTHORITATIVE = "authoritative"
    ENERGETIC = "energetic"
    CALM = "calm"


@dataclass
class VoiceCharacteristics:
    """Characteristics of a voice for matching."""
    voice_id: str
    gender: VoiceGender
    age: VoiceAge = VoiceAge.ADULT
    style: VoiceStyle = VoiceStyle.NEUTRAL
    accent: str = "american"
    quality_score: float = 1.0  # Higher = prefer this voice

    # Good for specific roles
    good_for_narrator: bool = False
    good_for_protagonist: bool = False
    good_for_villain: bool = False


# Voice characteristics for Kokoro voices
VOICE_CHARACTERISTICS: Dict[str, VoiceCharacteristics] = {
    # American Female voices
    "af_heart": VoiceCharacteristics(
        voice_id="af_heart",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.WARM,
        accent="american",
        quality_score=1.2,
        good_for_protagonist=True,
    ),
    "af_bella": VoiceCharacteristics(
        voice_id="af_bella",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.WARM,
        accent="american",
        quality_score=1.1,
        good_for_protagonist=True,
    ),
    "af_sarah": VoiceCharacteristics(
        voice_id="af_sarah",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.NEUTRAL,
        accent="american",
        quality_score=1.0,
    ),
    "af_nicole": VoiceCharacteristics(
        voice_id="af_nicole",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.ENERGETIC,
        accent="american",
        quality_score=1.0,
    ),
    "af_sky": VoiceCharacteristics(
        voice_id="af_sky",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.YOUNG,
        style=VoiceStyle.ENERGETIC,
        accent="american",
        quality_score=0.9,
    ),

    # American Male voices
    "am_michael": VoiceCharacteristics(
        voice_id="am_michael",
        gender=VoiceGender.MALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.NEUTRAL,
        accent="american",
        quality_score=1.1,
        good_for_protagonist=True,
    ),
    "am_adam": VoiceCharacteristics(
        voice_id="am_adam",
        gender=VoiceGender.MALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.AUTHORITATIVE,
        accent="american",
        quality_score=1.0,
        good_for_narrator=True,
    ),

    # British Male voices
    "bm_george": VoiceCharacteristics(
        voice_id="bm_george",
        gender=VoiceGender.MALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.AUTHORITATIVE,
        accent="british",
        quality_score=1.2,
        good_for_narrator=True,
    ),
    "bm_lewis": VoiceCharacteristics(
        voice_id="bm_lewis",
        gender=VoiceGender.MALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.CALM,
        accent="british",
        quality_score=1.0,
    ),
    "bm_daniel": VoiceCharacteristics(
        voice_id="bm_daniel",
        gender=VoiceGender.MALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.NEUTRAL,
        accent="british",
        quality_score=1.0,
        good_for_villain=True,
    ),

    # British Female voices
    "bf_emma": VoiceCharacteristics(
        voice_id="bf_emma",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.WARM,
        accent="british",
        quality_score=1.1,
        good_for_narrator=True,
    ),
    "bf_isabella": VoiceCharacteristics(
        voice_id="bf_isabella",
        gender=VoiceGender.FEMALE,
        age=VoiceAge.ADULT,
        style=VoiceStyle.AUTHORITATIVE,
        accent="british",
        quality_score=1.0,
    ),
}


# Role-based voice preferences
ROLE_PREFERENCES: Dict[str, Dict[str, any]] = {
    "narrator": {
        "style": VoiceStyle.AUTHORITATIVE,
        "prefer_british": True,
        "use_good_for_narrator": True,
    },
    "protagonist": {
        "style": VoiceStyle.WARM,
        "use_good_for_protagonist": True,
    },
    "villain": {
        "style": VoiceStyle.AUTHORITATIVE,
        "prefer_british": True,
        "use_good_for_villain": True,
    },
    "child": {
        "age": VoiceAge.YOUNG,
        "style": VoiceStyle.ENERGETIC,
    },
    "elder": {
        "age": VoiceAge.ELDERLY,
        "style": VoiceStyle.CALM,
    },
}


class VoiceAssigner:
    """
    Assigns voices to speakers based on characteristics and preferences.

    Ensures:
    - Voice diversity (no duplicate voices unless necessary)
    - Gender matching when hints are provided
    - Role-appropriate voice selection
    - Quality preference for main characters
    """

    def __init__(
        self,
        available_voices: Optional[Dict[str, VoiceCharacteristics]] = None,
        prefer_quality: bool = True,
        prefer_diversity: bool = True,
    ):
        """
        Initialize the voice assigner.

        Args:
            available_voices: Voice characteristics dict (defaults to VOICE_CHARACTERISTICS)
            prefer_quality: Prefer higher quality voices for main characters
            prefer_diversity: Try to avoid duplicate voice assignments
        """
        self.voices = available_voices or VOICE_CHARACTERISTICS
        self.prefer_quality = prefer_quality
        self.prefer_diversity = prefer_diversity
        self._assigned: Set[str] = set()

    def reset(self):
        """Reset assignment state for new script."""
        self._assigned = set()

    def assign_voices(
        self,
        script: ParsedScript,
        voice_overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Assign voices to all speakers in a script.

        Args:
            script: Parsed dialogue script
            voice_overrides: Manual voice assignments {speaker_name: voice_id}

        Returns:
            Dict mapping speaker names to voice IDs
        """
        self.reset()
        voice_overrides = voice_overrides or {}
        assignments: Dict[str, str] = {}

        # Sort speakers by importance (more lines = more important)
        speaker_line_counts = {}
        for line in script.lines:
            name = line.speaker.name
            speaker_line_counts[name] = speaker_line_counts.get(name, 0) + 1

        sorted_speakers = sorted(
            script.speakers.values(),
            key=lambda s: speaker_line_counts.get(s.name, 0),
            reverse=True,
        )

        # Assign voices
        for speaker in sorted_speakers:
            if speaker.name in voice_overrides:
                # Use manual override
                voice_id = voice_overrides[speaker.name]
                assignments[speaker.name] = voice_id
                self._assigned.add(voice_id)
            else:
                # Auto-assign based on characteristics
                voice_id = self._find_best_voice(speaker)
                assignments[speaker.name] = voice_id
                self._assigned.add(voice_id)

            # Update speaker object
            speaker.voice = assignments[speaker.name]

        return assignments

    def _find_best_voice(self, speaker: Speaker) -> str:
        """Find the best voice for a speaker based on characteristics."""
        candidates: List[Tuple[str, float]] = []

        for voice_id, chars in self.voices.items():
            score = self._calculate_match_score(speaker, chars)

            # Penalize already-assigned voices if preferring diversity
            if self.prefer_diversity and voice_id in self._assigned:
                score *= 0.3

            candidates.append((voice_id, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            return candidates[0][0]

        # Fallback to first available voice
        return list(self.voices.keys())[0] if self.voices else "af_heart"

    def _calculate_match_score(
        self,
        speaker: Speaker,
        voice: VoiceCharacteristics
    ) -> float:
        """Calculate how well a voice matches a speaker."""
        score = voice.quality_score if self.prefer_quality else 1.0

        # Gender matching
        if speaker.gender_hint:
            if speaker.gender_hint == "female" and voice.gender == VoiceGender.FEMALE:
                score += 2.0
            elif speaker.gender_hint == "male" and voice.gender == VoiceGender.MALE:
                score += 2.0
            elif speaker.gender_hint != voice.gender.value:
                score -= 3.0  # Strong penalty for gender mismatch

        # Role-based preferences
        role = speaker.name.lower()
        if role in ROLE_PREFERENCES:
            prefs = ROLE_PREFERENCES[role]

            if prefs.get("use_good_for_narrator") and voice.good_for_narrator:
                score += 1.5
            if prefs.get("use_good_for_protagonist") and voice.good_for_protagonist:
                score += 1.0
            if prefs.get("use_good_for_villain") and voice.good_for_villain:
                score += 1.0
            if prefs.get("prefer_british") and voice.accent == "british":
                score += 0.5
            if prefs.get("style") == voice.style:
                score += 0.5
            if prefs.get("age") == voice.age:
                score += 0.5

        # Accent matching
        if speaker.accent_hint:
            if speaker.accent_hint.lower() == voice.accent.lower():
                score += 0.5
            else:
                score -= 0.3

        return score

    def suggest_voices(
        self,
        speaker: Speaker,
        count: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Suggest top voices for a speaker.

        Args:
            speaker: Speaker to find voices for
            count: Number of suggestions to return

        Returns:
            List of (voice_id, score) tuples
        """
        candidates = []
        for voice_id, chars in self.voices.items():
            score = self._calculate_match_score(speaker, chars)
            candidates.append((voice_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:count]


def auto_assign_voices(
    script: ParsedScript,
    voice_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Convenience function to auto-assign voices to all speakers.

    Args:
        script: Parsed dialogue script
        voice_overrides: Manual voice assignments

    Returns:
        Dict mapping speaker names to voice IDs
    """
    assigner = VoiceAssigner()
    return assigner.assign_voices(script, voice_overrides)


def get_voice_for_gender(gender: str) -> str:
    """
    Get a default voice for a gender.

    Args:
        gender: "male", "female", or other

    Returns:
        Default voice ID
    """
    if gender.lower() == "female":
        return "af_heart"
    elif gender.lower() == "male":
        return "am_michael"
    else:
        return "bm_george"  # Narrator-style neutral


def list_voices_by_gender(gender: str) -> List[str]:
    """
    List all voices matching a gender.

    Args:
        gender: "male" or "female"

    Returns:
        List of voice IDs
    """
    target = VoiceGender.FEMALE if gender.lower() == "female" else VoiceGender.MALE
    return [
        v.voice_id for v in VOICE_CHARACTERISTICS.values()
        if v.gender == target
    ]
