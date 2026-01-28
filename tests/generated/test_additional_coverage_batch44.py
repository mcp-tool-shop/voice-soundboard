"""
Additional coverage tests - Batch 44: Voice Studio Features Coverage.

Comprehensive tests for:
- voice_soundboard/studio/session.py
- voice_soundboard/studio/engine.py
- voice_soundboard/studio/ai_assistant.py
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock


# =============================================================================
# VoiceStudioSession Tests
# =============================================================================

class TestVoiceStudioSession:
    """Tests for VoiceStudioSession class."""

    def test_create_new_session(self):
        """Test creating a new studio session."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        assert session is not None
        assert session.session_id is not None
        assert session.current_preset is not None

    def test_create_session_with_base_preset(self):
        """Test creating session from base preset."""
        from voice_soundboard.studio.session import VoiceStudioSession
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        base = VoicePreset(
            id="base:preset",
            name="Base Preset",
            source=PresetSource.CUSTOM,
            description="A base preset",
        )

        session = VoiceStudioSession.create_new(base_preset=base)
        assert session.base_preset_id == "base:preset"
        assert session.current_preset.name == "Base Preset"

    def test_create_session_with_custom_id(self):
        """Test creating session with custom ID."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new(session_id="custom_session_123")
        assert session.session_id == "custom_session_123"

    def test_apply_changes(self):
        """Test applying parameter changes."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        changes = session.apply_changes({"formant_ratio": 0.9})

        assert changes["new"]["formant_ratio"] == 0.9
        assert session.current_preset.acoustic.formant_ratio == 0.9

    def test_undo_changes(self):
        """Test undoing changes."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()

        # Apply change
        session.apply_changes({"formant_ratio": 0.9})
        assert session.current_preset.acoustic.formant_ratio == 0.9

        # Undo
        session.undo()
        assert session.current_preset.acoustic.formant_ratio == 1.0

    def test_redo_changes(self):
        """Test redoing changes."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()

        # Apply and undo
        session.apply_changes({"formant_ratio": 0.9})
        session.undo()
        assert session.current_preset.acoustic.formant_ratio == 1.0

        # Redo
        session.redo()
        assert session.current_preset.acoustic.formant_ratio == 0.9

    def test_undo_empty_stack(self):
        """Test undo with empty stack returns None."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        result = session.undo()
        assert result is None

    def test_redo_empty_stack(self):
        """Test redo with empty stack returns None."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        result = session.redo()
        assert result is None

    def test_max_undo_depth(self):
        """Test that undo stack respects max depth."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.max_undo_depth = 5

        # Apply more changes than max depth
        for i in range(10):
            session.apply_changes({"formant_ratio": 0.9 + i * 0.01})

        assert len(session.undo_stack) <= 5

    def test_new_change_clears_redo_stack(self):
        """Test that new changes clear the redo stack."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()

        # Apply, undo, then apply new
        session.apply_changes({"formant_ratio": 0.9})
        session.undo()
        assert len(session.redo_stack) == 1

        session.apply_changes({"formant_ratio": 0.85})
        assert len(session.redo_stack) == 0

    def test_get_current_params(self):
        """Test getting current parameters."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        params = session.get_current_params()

        assert "formant_ratio" in params
        assert "breath_intensity" in params
        assert "speed_factor" in params

    def test_get_preview_cache_key(self):
        """Test preview cache key generation."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        key1 = session.get_preview_cache_key()

        # Change params, key should change
        session.apply_changes({"formant_ratio": 0.9})
        key2 = session.get_preview_cache_key()

        assert key1 != key2

    def test_add_ai_suggestion(self):
        """Test adding AI suggestion."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.add_ai_suggestion(
            description="Make voice warmer",
            suggested_params={"formant_ratio": 0.95}
        )

        assert len(session.ai_suggestions) == 1
        assert session.ai_suggestions[0]["description"] == "Make voice warmer"
        assert session.last_description == "Make voice warmer"

    def test_mark_suggestion_applied(self):
        """Test marking suggestion as applied."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.add_ai_suggestion("Test", {"formant_ratio": 0.9})
        session.mark_suggestion_applied()

        assert session.ai_suggestions[-1]["applied"] is True

    def test_update_metadata(self):
        """Test updating preset metadata."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.update_metadata(
            name="Updated Name",
            description="Updated description",
            tags=["new", "tags"],
        )

        assert session.current_preset.name == "Updated Name"
        assert session.current_preset.description == "Updated description"
        assert "new" in session.current_preset.tags

    def test_update_metadata_with_enums(self):
        """Test updating metadata with enum fields."""
        from voice_soundboard.studio.session import VoiceStudioSession
        from voice_soundboard.presets.schema import Gender

        session = VoiceStudioSession.create_new()
        session.update_metadata(gender="female", energy="calm")

        assert session.current_preset.gender == Gender.FEMALE

    def test_to_dict(self):
        """Test serializing session to dict."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        data = session.to_dict()

        assert "session_id" in data
        assert "current_preset" in data
        assert "undo_depth" in data
        assert "created_at" in data

    def test_get_status(self):
        """Test getting session status."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.apply_changes({"formant_ratio": 0.9})

        status = session.get_status()
        assert "session_id" in status
        assert "parameters" in status
        assert status["can_undo"] is True
        assert status["can_redo"] is False


# =============================================================================
# Session Management Functions Tests
# =============================================================================

class TestSessionManagement:
    """Tests for session management functions."""

    def test_create_session(self):
        """Test create_session function."""
        from voice_soundboard.studio.session import create_session, get_current_session

        session = create_session()
        current = get_current_session()

        assert current is not None
        assert current.session_id == session.session_id

    def test_get_session(self):
        """Test get_session function."""
        from voice_soundboard.studio.session import create_session, get_session

        session = create_session()
        retrieved = get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_delete_session(self):
        """Test delete_session function."""
        from voice_soundboard.studio.session import (
            create_session, delete_session, get_session, get_current_session
        )

        session = create_session()
        session_id = session.session_id

        result = delete_session(session_id)
        assert result is True
        assert get_session(session_id) is None
        assert get_current_session() is None

    def test_list_sessions(self):
        """Test list_sessions function."""
        from voice_soundboard.studio.session import create_session, list_sessions

        # Create multiple sessions
        session1 = create_session()
        session2 = create_session()

        sessions = list_sessions()
        assert len(sessions) >= 2

        session_ids = [s["session_id"] for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    def test_set_current_session(self):
        """Test set_current_session function."""
        from voice_soundboard.studio.session import (
            create_session, set_current_session, get_current_session
        )

        session1 = create_session()
        session2 = create_session()

        # session2 is current after creation
        assert get_current_session().session_id == session2.session_id

        # Switch to session1
        set_current_session(session1.session_id)
        assert get_current_session().session_id == session1.session_id


# =============================================================================
# VoiceStudioEngine Tests
# =============================================================================

class TestVoiceStudioEngine:
    """Tests for VoiceStudioEngine class."""

    def test_engine_creation(self):
        """Test creating VoiceStudioEngine."""
        from voice_soundboard.studio.engine import VoiceStudioEngine

        engine = VoiceStudioEngine()
        assert engine is not None

    def test_apply_acoustic_params(self):
        """Test applying acoustic params to audio."""
        from voice_soundboard.studio.engine import VoiceStudioEngine
        from voice_soundboard.presets.schema import AcousticParams

        engine = VoiceStudioEngine()
        audio = np.random.randn(24000).astype(np.float32)
        params = AcousticParams(formant_ratio=1.0)  # No change

        result_audio, result_sr = engine.apply_acoustic_params(audio, params, 24000)
        assert result_audio is not None
        assert result_sr == 24000

    def test_apply_acoustic_params_with_formant(self):
        """Test applying formant shift."""
        from voice_soundboard.studio.engine import VoiceStudioEngine
        from voice_soundboard.presets.schema import AcousticParams

        engine = VoiceStudioEngine()
        audio = np.random.randn(24000).astype(np.float32)
        params = AcousticParams(formant_ratio=0.9)

        # Should not crash even if formant shifter not available
        result_audio, result_sr = engine.apply_acoustic_params(audio, params, 24000)
        assert result_audio is not None

    @pytest.mark.asyncio
    async def test_generate_preview_mock(self):
        """Test generate_preview with mocked engine."""
        from voice_soundboard.studio.engine import VoiceStudioEngine
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        engine = VoiceStudioEngine()

        preset = VoicePreset(
            id="test:preset",
            name="Test",
            source=PresetSource.CUSTOM,
            description="Test preset",
        )

        # Will use mock/silence since no real engine
        audio, sr = await engine.generate_preview(preset, "Hello", voice="af_bella")
        assert audio is not None
        assert sr > 0

    def test_generate_preview_sync(self):
        """Test synchronous preview generation."""
        from voice_soundboard.studio.engine import VoiceStudioEngine
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        engine = VoiceStudioEngine()

        preset = VoicePreset(
            id="sync:test",
            name="Sync Test",
            source=PresetSource.CUSTOM,
            description="Sync test preset",
        )

        audio, sr = engine.generate_preview_sync(preset, "Test")
        assert audio is not None
        assert sr > 0

    def test_save_preview(self, tmp_path):
        """Test saving preview audio."""
        from voice_soundboard.studio.engine import VoiceStudioEngine

        engine = VoiceStudioEngine()
        audio = np.random.randn(24000).astype(np.float32)

        path = engine.save_preview(audio, 24000, tmp_path / "test.wav")
        assert path.exists()


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestParameterValidation:
    """Tests for parameter validation functions."""

    def test_validate_params(self):
        """Test validate_params function."""
        from voice_soundboard.studio.engine import validate_params

        params = {
            "formant_ratio": 0.9,
            "breath_intensity": 0.2,
        }
        validated = validate_params(params)

        assert validated["formant_ratio"] == 0.9
        assert validated["breath_intensity"] == 0.2

    def test_validate_params_clamp(self):
        """Test that values are clamped to valid ranges."""
        from voice_soundboard.studio.engine import validate_params

        params = {
            "formant_ratio": 0.5,  # Below min (0.8)
            "speed_factor": 5.0,   # Above max (2.0)
        }
        validated = validate_params(params)

        assert validated["formant_ratio"] == 0.8  # Clamped to min
        assert validated["speed_factor"] == 2.0   # Clamped to max

    def test_validate_params_unknown(self):
        """Test that unknown params are ignored with warning."""
        from voice_soundboard.studio.engine import validate_params

        params = {
            "formant_ratio": 0.9,
            "unknown_param": 1.0,
        }
        validated = validate_params(params)

        assert "formant_ratio" in validated
        assert "unknown_param" not in validated


# =============================================================================
# Parameter Description Tests
# =============================================================================

class TestParamsToDescription:
    """Tests for params_to_description function."""

    def test_neutral_description(self):
        """Test neutral/balanced voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams()  # Default neutral
        desc = params_to_description(params)

        assert "neutral" in desc or "balanced" in desc

    def test_deep_voice_description(self):
        """Test deep voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(formant_ratio=0.85)
        desc = params_to_description(params)

        assert "deep" in desc.lower()

    def test_bright_voice_description(self):
        """Test bright voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(formant_ratio=1.15)
        desc = params_to_description(params)

        assert "bright" in desc.lower()

    def test_breathy_voice_description(self):
        """Test breathy voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(breath_intensity=0.3)
        desc = params_to_description(params)

        assert "breath" in desc.lower()

    def test_slow_voice_description(self):
        """Test slow voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(speed_factor=0.8)
        desc = params_to_description(params)

        assert "slow" in desc.lower() or "measured" in desc.lower()

    def test_fast_voice_description(self):
        """Test fast voice description."""
        from voice_soundboard.studio.engine import params_to_description
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(speed_factor=1.3)
        desc = params_to_description(params)

        assert "quick" in desc.lower() or "energetic" in desc.lower()


# =============================================================================
# Vocology Parameter Tests
# =============================================================================

class TestVocologyParams:
    """Tests for vocology-specific parameters."""

    def test_apply_vocology_params(self):
        """Test applying vocology parameters."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.apply_changes({
            "emotional_state": "excited",
            "scoop_cents": 40.0,
            "final_drop_cents": 30.0,
        })

        params = session.get_current_params()
        assert params["emotional_state"] == "excited"
        assert params["scoop_cents"] == 40.0

    def test_apply_phonation_params(self):
        """Test applying phonation parameters."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        session.apply_changes({
            "phonation_type": "breathy",
            "phonation_intensity": 0.7,
        })

        params = session.get_current_params()
        assert params["phonation_type"] == "breathy"
        assert params["phonation_intensity"] == 0.7


# =============================================================================
# Session Timestamps Tests
# =============================================================================

class TestSessionTimestamps:
    """Tests for session timestamp tracking."""

    def test_created_at_timestamp(self):
        """Test created_at timestamp."""
        from voice_soundboard.studio.session import VoiceStudioSession

        session = VoiceStudioSession.create_new()
        assert session.created_at is not None
        assert isinstance(session.created_at, datetime)

    def test_modified_at_updates(self):
        """Test that modified_at updates on changes."""
        from voice_soundboard.studio.session import VoiceStudioSession
        import time

        session = VoiceStudioSession.create_new()
        original_modified = session.modified_at

        time.sleep(0.01)  # Small delay
        session.apply_changes({"formant_ratio": 0.9})

        assert session.modified_at > original_modified


# =============================================================================
# Parameter Ranges Tests
# =============================================================================

class TestParamRanges:
    """Tests for PARAM_RANGES constant."""

    def test_param_ranges_exist(self):
        """Test that PARAM_RANGES has expected parameters."""
        from voice_soundboard.studio.engine import PARAM_RANGES

        expected_params = [
            "formant_ratio",
            "breath_intensity",
            "speed_factor",
            "pitch_shift_semitones",
        ]

        for param in expected_params:
            assert param in PARAM_RANGES

    def test_param_ranges_valid_tuples(self):
        """Test that all ranges are valid tuples."""
        from voice_soundboard.studio.engine import PARAM_RANGES

        for param, range_tuple in PARAM_RANGES.items():
            assert isinstance(range_tuple, tuple)
            assert len(range_tuple) == 2
            assert range_tuple[0] < range_tuple[1]  # Min < Max
