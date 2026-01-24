"""
Tests for the Voice Studio module.

Tests cover:
- VoiceStudioSession state management
- VoiceStudioEngine preview generation
- StudioAIAssistant natural language parsing
- Undo/redo functionality
- Session lifecycle
"""

import pytest
import numpy as np
from pathlib import Path

from voice_soundboard.studio import (
    VoiceStudioSession,
    VoiceStudioEngine,
    StudioAIAssistant,
)
from voice_soundboard.studio.session import (
    create_session,
    get_session,
    get_current_session,
    set_current_session,
    delete_session,
    list_sessions,
    _sessions,
)
from voice_soundboard.studio.engine import (
    validate_params,
    params_to_description,
    PARAM_RANGES,
)
from voice_soundboard.presets import VoicePreset, PresetSource, AcousticParams


class TestVoiceStudioSession:
    """Tests for VoiceStudioSession class."""

    def test_create_new_session(self):
        """Can create a new session from scratch."""
        session = VoiceStudioSession.create_new()

        assert session.session_id.startswith("studio_")
        assert session.current_preset is not None
        assert session.current_preset.name == "New Preset"
        assert session.current_preset.source == PresetSource.CUSTOM
        assert session.base_preset_id is None
        assert len(session.undo_stack) == 0
        assert len(session.redo_stack) == 0

    def test_create_session_with_base_preset(self):
        """Can create a session based on existing preset."""
        base_preset = VoicePreset(
            id="test:base",
            name="Test Base",
            source=PresetSource.VOCOLOGY,
            description="Base preset for testing",
            acoustic=AcousticParams(formant_ratio=0.95),
        )

        session = VoiceStudioSession.create_new(base_preset=base_preset)

        assert session.base_preset_id == "test:base"
        assert session.current_preset.id.startswith("studio:")
        assert session.current_preset.source == PresetSource.CUSTOM
        # Acoustic params should be copied
        assert session.current_preset.acoustic.formant_ratio == 0.95

    def test_create_session_with_custom_id(self):
        """Can create a session with a custom ID."""
        session = VoiceStudioSession.create_new(session_id="my_custom_session")

        assert session.session_id == "my_custom_session"

    def test_apply_changes(self):
        """Can apply parameter changes."""
        session = VoiceStudioSession.create_new()

        changes = session.apply_changes({"formant_ratio": 0.9, "breath_intensity": 0.2})

        assert changes["new"]["formant_ratio"] == 0.9
        assert changes["new"]["breath_intensity"] == 0.2
        assert session.current_preset.acoustic.formant_ratio == 0.9
        assert session.current_preset.acoustic.breath_intensity == 0.2

    def test_apply_changes_creates_undo_state(self):
        """Applying changes pushes state to undo stack."""
        session = VoiceStudioSession.create_new()
        assert len(session.undo_stack) == 0

        session.apply_changes({"formant_ratio": 0.9})

        assert len(session.undo_stack) == 1

    def test_apply_changes_clears_redo_stack(self):
        """Applying new changes clears redo stack."""
        session = VoiceStudioSession.create_new()
        session.apply_changes({"formant_ratio": 0.9})
        session.undo()
        assert len(session.redo_stack) == 1

        session.apply_changes({"formant_ratio": 0.85})

        assert len(session.redo_stack) == 0

    def test_undo(self):
        """Can undo parameter changes."""
        session = VoiceStudioSession.create_new()
        original_ratio = session.current_preset.acoustic.formant_ratio

        session.apply_changes({"formant_ratio": 0.9})
        assert session.current_preset.acoustic.formant_ratio == 0.9

        state = session.undo()

        assert state is not None
        assert session.current_preset.acoustic.formant_ratio == original_ratio

    def test_undo_empty_stack_returns_none(self):
        """Undo with empty stack returns None."""
        session = VoiceStudioSession.create_new()

        result = session.undo()

        assert result is None

    def test_redo(self):
        """Can redo undone changes."""
        session = VoiceStudioSession.create_new()
        session.apply_changes({"formant_ratio": 0.9})
        session.undo()

        state = session.redo()

        assert state is not None
        assert session.current_preset.acoustic.formant_ratio == 0.9

    def test_redo_empty_stack_returns_none(self):
        """Redo with empty stack returns None."""
        session = VoiceStudioSession.create_new()

        result = session.redo()

        assert result is None

    def test_multiple_undo_redo(self):
        """Multiple undo/redo operations work correctly."""
        session = VoiceStudioSession.create_new()

        session.apply_changes({"formant_ratio": 0.9})
        session.apply_changes({"formant_ratio": 0.85})
        session.apply_changes({"formant_ratio": 0.8})

        assert session.current_preset.acoustic.formant_ratio == 0.8

        session.undo()
        assert session.current_preset.acoustic.formant_ratio == 0.85

        session.undo()
        assert session.current_preset.acoustic.formant_ratio == 0.9

        session.redo()
        assert session.current_preset.acoustic.formant_ratio == 0.85

    def test_max_undo_depth(self):
        """Undo stack respects max depth."""
        session = VoiceStudioSession.create_new()
        session.max_undo_depth = 5

        for i in range(10):
            session.apply_changes({"formant_ratio": 0.9 - i * 0.01})

        assert len(session.undo_stack) == 5

    def test_get_current_params(self):
        """Can get current params as dictionary."""
        session = VoiceStudioSession.create_new()
        session.apply_changes({"formant_ratio": 0.9, "jitter_percent": 1.0})

        params = session.get_current_params()

        assert params["formant_ratio"] == 0.9
        assert params["jitter_percent"] == 1.0

    def test_get_preview_cache_key(self):
        """Cache key changes when parameters change."""
        session = VoiceStudioSession.create_new()

        key1 = session.get_preview_cache_key()
        session.apply_changes({"formant_ratio": 0.9})
        key2 = session.get_preview_cache_key()

        assert key1 != key2

    def test_ai_suggestion_tracking(self):
        """AI suggestions are tracked."""
        session = VoiceStudioSession.create_new()

        session.add_ai_suggestion("warm narrator", {"formant_ratio": 0.95})

        assert len(session.ai_suggestions) == 1
        assert session.ai_suggestions[0]["description"] == "warm narrator"
        assert session.ai_suggestions[0]["applied"] == False
        assert session.last_description == "warm narrator"

    def test_mark_suggestion_applied(self):
        """Can mark suggestion as applied."""
        session = VoiceStudioSession.create_new()
        session.add_ai_suggestion("deep voice", {"formant_ratio": 0.9})

        session.mark_suggestion_applied()

        assert session.ai_suggestions[0]["applied"] == True

    def test_update_metadata(self):
        """Can update preset metadata."""
        session = VoiceStudioSession.create_new()

        session.update_metadata(
            name="My Custom Voice",
            description="A custom voice preset",
            tags=["warm", "narrator"],
        )

        assert session.current_preset.name == "My Custom Voice"
        assert session.current_preset.description == "A custom voice preset"
        assert "warm" in session.current_preset.tags

    def test_to_dict(self):
        """Session can be serialized to dict."""
        session = VoiceStudioSession.create_new()

        data = session.to_dict()

        assert "session_id" in data
        assert "current_preset" in data
        assert "undo_depth" in data
        assert "created_at" in data

    def test_get_status(self):
        """Can get session status."""
        session = VoiceStudioSession.create_new()
        session.apply_changes({"formant_ratio": 0.9})

        status = session.get_status()

        assert "session_id" in status
        assert "parameters" in status
        assert "can_undo" in status
        assert status["can_undo"] == True


class TestSessionManagement:
    """Tests for global session management."""

    def setup_method(self):
        """Clear sessions before each test."""
        _sessions.clear()

    def test_create_session_function(self):
        """create_session creates and registers a session."""
        session = create_session()

        assert session is not None
        assert session.session_id in _sessions
        assert get_current_session() == session

    def test_get_session_by_id(self):
        """Can retrieve session by ID."""
        session = create_session()

        retrieved = get_session(session.session_id)

        assert retrieved == session

    def test_get_session_nonexistent(self):
        """Getting nonexistent session returns None."""
        result = get_session("nonexistent")

        assert result is None

    def test_set_current_session(self):
        """Can set current session."""
        session1 = create_session()
        session2 = create_session()

        set_current_session(session1.session_id)

        assert get_current_session() == session1

    def test_delete_session(self):
        """Can delete a session."""
        session = create_session()
        session_id = session.session_id

        result = delete_session(session_id)

        assert result == True
        assert get_session(session_id) is None

    def test_delete_current_session_clears_current(self):
        """Deleting current session clears current reference."""
        session = create_session()

        delete_session(session.session_id)

        assert get_current_session() is None

    def test_list_sessions(self):
        """Can list all sessions."""
        create_session()
        create_session()

        sessions = list_sessions()

        assert len(sessions) == 2
        assert all("session_id" in s for s in sessions)


class TestStudioAIAssistant:
    """Tests for StudioAIAssistant class."""

    @pytest.fixture
    def ai(self):
        return StudioAIAssistant()

    def test_parse_deep_voice(self, ai):
        """Parses 'deep' voice description."""
        result = ai.parse_description("deep voice")

        assert "formant_ratio" in result["params"]
        assert result["params"]["formant_ratio"] < 1.0
        assert "deep" in result["matched_keywords"]

    def test_parse_bright_voice(self, ai):
        """Parses 'bright' voice description."""
        result = ai.parse_description("bright voice")

        assert "formant_ratio" in result["params"]
        assert result["params"]["formant_ratio"] > 1.0
        assert "bright" in result["matched_keywords"]

    def test_parse_warm_narrator(self, ai):
        """Parses 'warm narrator' description."""
        result = ai.parse_description("warm narrator voice")

        assert "formant_ratio" in result["params"]
        assert "warm" in result["matched_keywords"] or "narrator" in result["matched_keywords"]

    def test_parse_breathy(self, ai):
        """Parses 'breathy' description."""
        result = ai.parse_description("breathy voice")

        assert "breath_intensity" in result["params"]
        assert result["params"]["breath_intensity"] > 0.2
        assert "breathy" in result["matched_keywords"]

    def test_parse_gravelly(self, ai):
        """Parses 'gravelly' description."""
        result = ai.parse_description("gravelly voice")

        assert "jitter_percent" in result["params"]
        assert result["params"]["jitter_percent"] > 1.0
        assert "gravelly" in result["matched_keywords"]

    def test_parse_modifier_very(self, ai):
        """Modifiers scale parameters."""
        result1 = ai.parse_description("deep voice")
        result2 = ai.parse_description("very deep voice")

        # "very" should make it deeper (lower formant ratio)
        assert result2["params"]["formant_ratio"] < result1["params"]["formant_ratio"]

    def test_parse_modifier_slightly(self, ai):
        """'slightly' modifier reduces intensity."""
        result1 = ai.parse_description("deep voice")
        result2 = ai.parse_description("slightly deep voice")

        # "slightly" should be closer to 1.0
        diff1 = abs(1.0 - result1["params"]["formant_ratio"])
        diff2 = abs(1.0 - result2["params"]["formant_ratio"])
        assert diff2 < diff1

    def test_parse_combined_qualities(self, ai):
        """Can parse multiple qualities."""
        result = ai.parse_description("warm breathy narrator")

        assert len(result["matched_keywords"]) >= 2

    def test_parse_returns_suggestions(self, ai):
        """Parse returns textual suggestions."""
        result = ai.parse_description("deep voice")

        assert "suggestions" in result
        assert len(result["suggestions"]) > 0

    def test_parse_no_match(self, ai):
        """Returns empty params for unrecognized description."""
        result = ai.parse_description("xyz123 qwerty")

        assert len(result["matched_keywords"]) == 0

    def test_get_qualities_list(self, ai):
        """Can get list of available qualities."""
        qualities = ai.get_qualities_list()

        assert "deep" in qualities
        assert "warm" in qualities
        assert "bright" in qualities

    def test_suggest_from_params(self, ai):
        """Can suggest description from params."""
        params = AcousticParams(formant_ratio=0.88, jitter_percent=2.0)

        description = ai.suggest_from_params(params)

        assert isinstance(description, str)
        assert len(description) > 0


class TestVoiceStudioEngine:
    """Tests for VoiceStudioEngine class."""

    @pytest.fixture
    def engine(self):
        return VoiceStudioEngine()

    def test_engine_initialization(self, engine):
        """Engine initializes with lazy loading."""
        assert engine._voice_engine is None
        assert engine._formant_shifter is None
        assert engine._humanizer is None

    def test_apply_acoustic_params(self, engine):
        """Can apply acoustic params to audio."""
        # Create test audio
        sr = 24000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float32)
        params = AcousticParams(formant_ratio=0.95)

        result_audio, result_sr = engine.apply_acoustic_params(audio, params, sr)

        assert isinstance(result_audio, np.ndarray)
        assert result_sr == sr


class TestEngineHelpers:
    """Tests for engine helper functions."""

    def test_validate_params_clamps_values(self):
        """validate_params clamps to valid ranges."""
        params = {"formant_ratio": 0.5, "jitter_percent": 10.0}

        validated = validate_params(params)

        min_f, max_f = PARAM_RANGES["formant_ratio"]
        min_j, max_j = PARAM_RANGES["jitter_percent"]
        assert validated["formant_ratio"] == min_f
        assert validated["jitter_percent"] == max_j

    def test_validate_params_ignores_unknown(self):
        """validate_params ignores unknown parameters."""
        params = {"unknown_param": 1.0, "formant_ratio": 0.95}

        validated = validate_params(params)

        assert "unknown_param" not in validated
        assert "formant_ratio" in validated

    def test_params_to_description_deep(self):
        """params_to_description describes deep voice."""
        params = AcousticParams(formant_ratio=0.88)

        desc = params_to_description(params)

        assert "deep" in desc.lower()

    def test_params_to_description_bright(self):
        """params_to_description describes bright voice."""
        params = AcousticParams(formant_ratio=1.12)

        desc = params_to_description(params)

        assert "bright" in desc.lower()

    def test_params_to_description_breathy(self):
        """params_to_description describes breathy voice."""
        params = AcousticParams(breath_intensity=0.3)

        desc = params_to_description(params)

        assert "breathy" in desc.lower()

    def test_params_to_description_neutral(self):
        """params_to_description handles neutral params."""
        params = AcousticParams()

        desc = params_to_description(params)

        assert "neutral" in desc.lower() or "balanced" in desc.lower()


class TestIntegration:
    """Integration tests for the complete studio workflow."""

    def setup_method(self):
        _sessions.clear()

    def test_complete_workflow(self):
        """Test complete preset creation workflow."""
        # 1. Start session
        session = create_session()
        assert session is not None

        # 2. Parse AI description
        ai = StudioAIAssistant()
        result = ai.parse_description("warm deep narrator")

        # 3. Apply parameters
        if result["params"]:
            session.apply_changes(result["params"])
            session.add_ai_suggestion("warm deep narrator", result["params"])
            session.mark_suggestion_applied()

        # 4. Adjust manually
        session.apply_changes({"breath_intensity": 0.2})

        # 5. Undo
        session.undo()
        assert session.current_preset.acoustic.breath_intensity != 0.2

        # 6. Redo
        session.redo()
        assert session.current_preset.acoustic.breath_intensity == 0.2

        # 7. Update metadata
        session.update_metadata(
            name="My Narrator",
            description="Custom warm narrator voice",
            tags=["narrator", "warm", "custom"],
        )

        # 8. Verify final state
        assert session.current_preset.name == "My Narrator"
        assert len(session.ai_suggestions) == 1
        assert session.ai_suggestions[0]["applied"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
