"""
Voice Studio Session Management

Manages the state of a voice studio editing session, including:
- Current preset being edited
- Undo/redo stack for parameter changes
- Preview configuration
- AI suggestion history
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import copy
import hashlib
import json

from voice_soundboard.presets.schema import (
    VoicePreset,
    AcousticParams,
    PresetSource,
    Gender,
    AgeRange,
    VoiceEnergy,
    VoiceTone,
)


@dataclass
class VoiceStudioSession:
    """
    Manages state for a voice studio editing session.

    Provides undo/redo functionality, preview caching, and
    tracks AI-suggested changes.

    Example:
        >>> session = VoiceStudioSession.create_new()
        >>> session.apply_changes({"formant_ratio": 0.9})
        >>> session.undo()  # Restores previous state
    """

    session_id: str
    current_preset: VoicePreset
    base_preset_id: Optional[str] = None

    # State management
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)
    max_undo_depth: int = 20

    # Preview configuration
    preview_text: str = "Hello! This is a voice preview sample. How does this sound to you?"
    preview_voice: str = "af_bella"

    # AI assistance tracking
    ai_suggestions: list[dict] = field(default_factory=list)
    last_description: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create_new(
        cls,
        session_id: Optional[str] = None,
        base_preset: Optional[VoicePreset] = None,
        preview_text: Optional[str] = None,
        preview_voice: Optional[str] = None,
    ) -> "VoiceStudioSession":
        """
        Create a new studio session.

        Args:
            session_id: Optional custom session ID
            base_preset: Optional preset to use as starting point
            preview_text: Custom preview text
            preview_voice: Kokoro voice ID for preview synthesis

        Returns:
            New VoiceStudioSession instance
        """
        import time
        import random

        if session_id is None:
            session_id = f"studio_{int(time.time())}_{random.randint(1000, 9999)}"

        if base_preset:
            # Create a working copy
            preset_dict = base_preset.to_dict()
            preset_dict["id"] = f"studio:{session_id}"
            preset_dict["source"] = "custom"
            current_preset = VoicePreset.from_dict(preset_dict)
            base_preset_id = base_preset.id
        else:
            # Start from scratch with neutral parameters
            current_preset = VoicePreset(
                id=f"studio:{session_id}",
                name="New Preset",
                source=PresetSource.CUSTOM,
                description="Custom voice preset created in Voice Studio",
                acoustic=AcousticParams(),
                tags=[],
                use_cases=[],
            )
            base_preset_id = None

        return cls(
            session_id=session_id,
            current_preset=current_preset,
            base_preset_id=base_preset_id,
            preview_text=preview_text or cls.preview_text,
            preview_voice=preview_voice or cls.preview_voice,
        )

    def _ensure_acoustic_params(self) -> AcousticParams:
        """Ensure current preset has acoustic params."""
        if self.current_preset.acoustic is None:
            self.current_preset.acoustic = AcousticParams()
        return self.current_preset.acoustic

    def _save_state(self) -> dict:
        """Save current state for undo stack."""
        return {
            "preset_dict": self.current_preset.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }

    def _restore_state(self, state: dict) -> None:
        """Restore state from undo/redo stack."""
        self.current_preset = VoicePreset.from_dict(state["preset_dict"])
        self.modified_at = datetime.now()

    def apply_changes(self, params: dict[str, float]) -> dict:
        """
        Apply parameter changes to current preset.

        Pushes current state to undo stack before applying.

        Args:
            params: Dictionary of parameter name -> value
                Valid keys: formant_ratio, breath_intensity, jitter_percent,
                shimmer_percent, pitch_drift_cents, timing_variation_ms,
                speed_factor, pitch_shift_semitones, breath_volume_db

        Returns:
            Dictionary with old and new values
        """
        # Save current state for undo
        self.undo_stack.append(self._save_state())
        if len(self.undo_stack) > self.max_undo_depth:
            self.undo_stack.pop(0)

        # Clear redo stack on new change
        self.redo_stack.clear()

        # Get or create acoustic params
        acoustic = self._ensure_acoustic_params()

        # Track changes
        changes = {"old": {}, "new": {}}

        # Apply each parameter
        param_mapping = {
            # Basic params
            "formant_ratio": "formant_ratio",
            "breath_intensity": "breath_intensity",
            "breath_volume_db": "breath_volume_db",
            "jitter_percent": "jitter_percent",
            "shimmer_percent": "shimmer_percent",
            "pitch_drift_cents": "pitch_drift_cents",
            "timing_variation_ms": "timing_variation_ms",
            "speed_factor": "speed_factor",
            "pitch_shift_semitones": "pitch_shift_semitones",
            # Vocology params
            "emotional_state": "emotional_state",
            "scoop_cents": "scoop_cents",
            "final_drop_cents": "final_drop_cents",
            "overshoot_cents": "overshoot_cents",
            "timing_bias_ms": "timing_bias_ms",
            # Research Lab params
            "phonation_type": "phonation_type",
            "phonation_intensity": "phonation_intensity",
            "jitter_rate_hz": "jitter_rate_hz",
            "drift_rate_hz": "drift_rate_hz",
            "iu_duration_s": "iu_duration_s",
            "hnr_target_db": "hnr_target_db",
            "spectral_tilt_db": "spectral_tilt_db",
        }

        for key, value in params.items():
            if key in param_mapping:
                attr_name = param_mapping[key]
                changes["old"][key] = getattr(acoustic, attr_name)
                setattr(acoustic, attr_name, value)
                changes["new"][key] = value

        self.modified_at = datetime.now()
        return changes

    def undo(self) -> Optional[dict]:
        """
        Undo the last parameter change.

        Returns:
            The restored state dict, or None if nothing to undo
        """
        if not self.undo_stack:
            return None

        # Save current state to redo stack
        self.redo_stack.append(self._save_state())

        # Restore previous state
        state = self.undo_stack.pop()
        self._restore_state(state)

        return state

    def redo(self) -> Optional[dict]:
        """
        Redo a previously undone change.

        Returns:
            The restored state dict, or None if nothing to redo
        """
        if not self.redo_stack:
            return None

        # Save current state to undo stack
        self.undo_stack.append(self._save_state())

        # Restore redo state
        state = self.redo_stack.pop()
        self._restore_state(state)

        return state

    def get_current_params(self) -> dict:
        """Get current acoustic parameters as dictionary."""
        acoustic = self._ensure_acoustic_params()
        return acoustic.to_dict()

    def get_preview_cache_key(self) -> str:
        """
        Generate a cache key based on current parameters.

        Used for caching preview audio to avoid re-synthesis
        when parameters haven't changed.
        """
        params = self.get_current_params()
        params_str = json.dumps(params, sort_keys=True)
        key_data = f"{self.preview_text}:{self.preview_voice}:{params_str}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    def add_ai_suggestion(self, description: str, suggested_params: dict) -> None:
        """Track an AI suggestion for history."""
        self.ai_suggestions.append({
            "description": description,
            "params": suggested_params,
            "timestamp": datetime.now().isoformat(),
            "applied": False,
        })
        self.last_description = description

    def mark_suggestion_applied(self, index: int = -1) -> None:
        """Mark a suggestion as applied."""
        if self.ai_suggestions:
            self.ai_suggestions[index]["applied"] = True

    def update_metadata(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_cases: Optional[list[str]] = None,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        accent: Optional[str] = None,
        energy: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> None:
        """Update preset metadata fields."""
        if name is not None:
            self.current_preset.name = name
        if description is not None:
            self.current_preset.description = description
        if tags is not None:
            self.current_preset.tags = tags
        if use_cases is not None:
            self.current_preset.use_cases = use_cases
        if gender is not None:
            self.current_preset.gender = Gender(gender) if gender else None
        if age_range is not None:
            self.current_preset.age_range = AgeRange(age_range) if age_range else None
        if accent is not None:
            self.current_preset.accent = accent
        if energy is not None:
            self.current_preset.energy = VoiceEnergy(energy) if energy else VoiceEnergy.NEUTRAL
        if tone is not None:
            self.current_preset.tone = VoiceTone(tone) if tone else VoiceTone.NEUTRAL

        self.modified_at = datetime.now()

    def to_dict(self) -> dict:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "current_preset": self.current_preset.to_dict(),
            "base_preset_id": self.base_preset_id,
            "preview_text": self.preview_text,
            "preview_voice": self.preview_voice,
            "undo_depth": len(self.undo_stack),
            "redo_depth": len(self.redo_stack),
            "ai_suggestions_count": len(self.ai_suggestions),
            "last_description": self.last_description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    def get_status(self) -> dict:
        """Get session status for display."""
        params = self.get_current_params()
        return {
            "session_id": self.session_id,
            "base_preset": self.base_preset_id,
            "name": self.current_preset.name,
            "description": self.current_preset.description,
            "parameters": params,  # Return all parameters (now includes vocology)
            "can_undo": len(self.undo_stack) > 0,
            "can_redo": len(self.redo_stack) > 0,
            "preview_voice": self.preview_voice,
        }


# Global session storage
_sessions: dict[str, VoiceStudioSession] = {}
_current_session_id: Optional[str] = None


def get_session(session_id: str) -> Optional[VoiceStudioSession]:
    """Get a session by ID."""
    return _sessions.get(session_id)


def get_current_session() -> Optional[VoiceStudioSession]:
    """Get the current active session."""
    if _current_session_id:
        return _sessions.get(_current_session_id)
    return None


def set_current_session(session_id: str) -> None:
    """Set the current active session."""
    global _current_session_id
    _current_session_id = session_id


def create_session(
    base_preset: Optional[VoicePreset] = None,
    preview_text: Optional[str] = None,
    preview_voice: Optional[str] = None,
) -> VoiceStudioSession:
    """Create and register a new session."""
    global _current_session_id

    session = VoiceStudioSession.create_new(
        base_preset=base_preset,
        preview_text=preview_text,
        preview_voice=preview_voice,
    )

    _sessions[session.session_id] = session
    _current_session_id = session.session_id

    return session


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    global _current_session_id

    if session_id in _sessions:
        del _sessions[session_id]
        if _current_session_id == session_id:
            _current_session_id = None
        return True
    return False


def list_sessions() -> list[dict]:
    """List all active sessions."""
    return [
        {
            "session_id": s.session_id,
            "name": s.current_preset.name,
            "base_preset": s.base_preset_id,
            "created_at": s.created_at.isoformat(),
            "modified_at": s.modified_at.isoformat(),
            "is_current": s.session_id == _current_session_id,
        }
        for s in _sessions.values()
    ]
