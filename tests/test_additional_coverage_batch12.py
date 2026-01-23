"""
Additional test coverage batch 12: emotion/curves.py, cloning/extractor.py, cloning/separation.py.

Tests for dynamic emotion curves, voice embedding extraction, and emotion-timbre separation.
"""

import pytest
import math
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np

from voice_soundboard.emotion.curves import (
    EmotionKeyframe,
    EmotionCurve,
    create_linear_curve,
    create_arc_curve,
    create_buildup_curve,
    create_fade_curve,
    create_wave_curve,
    NARRATIVE_CURVES,
    get_narrative_curve,
    list_narrative_curves,
)
from voice_soundboard.emotion.vad import VADPoint, VAD_EMOTIONS

from voice_soundboard.cloning.extractor import (
    ExtractorBackend,
    VoiceEmbedding,
    VoiceExtractor,
    extract_embedding,
)
from voice_soundboard.cloning.separation import (
    EmotionStyle,
    TimbreEmbedding,
    EmotionEmbedding,
    SeparatedVoice,
    EmotionTimbreSeparator,
    separate_voice,
    transfer_emotion,
)


# =============================================================================
# EmotionKeyframe Tests
# =============================================================================

class TestEmotionKeyframe:
    """Tests for EmotionKeyframe dataclass."""

    def test_creation_basic(self):
        """Test basic keyframe creation."""
        kf = EmotionKeyframe(position=0.5, emotion="happy")
        assert kf.position == 0.5
        assert kf.emotion == "happy"
        assert kf.intensity == 1.0
        assert kf.easing == "linear"

    def test_vad_computed_automatically(self):
        """Test VAD is computed from emotion name."""
        kf = EmotionKeyframe(position=0.0, emotion="happy")
        assert kf.vad is not None
        assert isinstance(kf.vad, VADPoint)

    def test_position_clamped_high(self):
        """Test position clamped to max 1.0."""
        kf = EmotionKeyframe(position=1.5, emotion="happy")
        assert kf.position == 1.0

    def test_position_clamped_low(self):
        """Test position clamped to min 0.0."""
        kf = EmotionKeyframe(position=-0.5, emotion="happy")
        assert kf.position == 0.0

    def test_unknown_emotion_falls_back_to_neutral(self):
        """Test unknown emotion uses neutral VAD."""
        kf = EmotionKeyframe(position=0.5, emotion="unknownemotionxyz")
        assert kf.vad == VAD_EMOTIONS["neutral"]

    def test_custom_easing(self):
        """Test custom easing setting."""
        kf = EmotionKeyframe(position=0.5, emotion="happy", easing="ease_in")
        assert kf.easing == "ease_in"

    def test_custom_intensity(self):
        """Test custom intensity setting."""
        kf = EmotionKeyframe(position=0.5, emotion="happy", intensity=0.7)
        assert kf.intensity == 0.7


# =============================================================================
# EmotionCurve Tests
# =============================================================================

class TestEmotionCurve:
    """Tests for EmotionCurve class."""

    def test_init_default(self):
        """Test default initialization."""
        curve = EmotionCurve()
        assert curve.default_easing == "linear"
        assert len(curve.keyframes) == 0

    def test_init_custom_easing(self):
        """Test custom default easing."""
        curve = EmotionCurve(default_easing="ease_in")
        assert curve.default_easing == "ease_in"

    def test_add_point_basic(self):
        """Test adding a point."""
        curve = EmotionCurve()
        result = curve.add_point(0.5, "happy")
        assert len(curve.keyframes) == 1
        assert curve.keyframes[0].position == 0.5
        assert result is curve  # Chaining

    def test_add_point_sorted(self):
        """Test points are sorted by position."""
        curve = EmotionCurve()
        curve.add_point(0.8, "happy")
        curve.add_point(0.2, "sad")
        curve.add_point(0.5, "neutral")
        positions = [kf.position for kf in curve.keyframes]
        assert positions == [0.2, 0.5, 0.8]

    def test_remove_point_exists(self):
        """Test removing existing point."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        result = curve.remove_point(0.5)
        assert result is True
        assert len(curve.keyframes) == 0

    def test_remove_point_not_found(self):
        """Test removing non-existent point."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        result = curve.remove_point(0.8)
        assert result is False
        assert len(curve.keyframes) == 1

    def test_remove_point_with_tolerance(self):
        """Test removing with tolerance."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        result = curve.remove_point(0.505, tolerance=0.01)
        assert result is True

    def test_clear(self):
        """Test clearing all keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        result = curve.clear()
        assert len(curve.keyframes) == 0
        assert result is curve

    def test_get_vad_at_empty(self):
        """Test VAD at position with empty curve."""
        curve = EmotionCurve()
        vad = curve.get_vad_at(0.5)
        assert vad == VAD_EMOTIONS["neutral"]

    def test_get_vad_at_single_keyframe(self):
        """Test VAD with single keyframe."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy")
        vad = curve.get_vad_at(0.0)
        # Should return the single keyframe's VAD
        assert isinstance(vad, VADPoint)

    def test_get_vad_at_interpolated(self):
        """Test VAD interpolation between keyframes."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        vad = curve.get_vad_at(0.5)
        # Should be interpolated between happy and sad
        assert isinstance(vad, VADPoint)

    def test_get_vad_at_clamped(self):
        """Test VAD position clamping."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        # Positions outside 0-1 should be clamped
        vad_low = curve.get_vad_at(-0.5)
        vad_high = curve.get_vad_at(1.5)
        assert isinstance(vad_low, VADPoint)
        assert isinstance(vad_high, VADPoint)

    def test_get_emotion_at(self):
        """Test getting emotion name at position."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        emotion = curve.get_emotion_at(0.5)
        assert isinstance(emotion, str)

    def test_get_intensity_at_empty(self):
        """Test intensity with empty curve."""
        curve = EmotionCurve()
        intensity = curve.get_intensity_at(0.5)
        assert intensity == 1.0

    def test_get_intensity_at_single(self):
        """Test intensity with single keyframe."""
        curve = EmotionCurve()
        curve.add_point(0.5, "happy", intensity=0.8)
        intensity = curve.get_intensity_at(0.5)
        assert intensity == 0.8

    def test_get_intensity_at_interpolated(self):
        """Test intensity interpolation."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy", intensity=0.0)
        curve.add_point(1.0, "sad", intensity=1.0)
        intensity = curve.get_intensity_at(0.5)
        assert abs(intensity - 0.5) < 0.01

    def test_sample(self):
        """Test sampling the curve."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        curve.add_point(1.0, "sad")
        samples = curve.sample(num_samples=5)
        assert len(samples) == 5
        assert samples[0][0] == 0.0
        assert samples[-1][0] == 1.0

    def test_sample_single(self):
        """Test sampling with single sample."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy")
        samples = curve.sample(num_samples=1)
        assert len(samples) == 1

    def test_to_keyframes_dict(self):
        """Test serialization to dict."""
        curve = EmotionCurve()
        curve.add_point(0.0, "happy", intensity=0.8, easing="ease_in")
        curve.add_point(1.0, "sad")
        data = curve.to_keyframes_dict()
        assert len(data) == 2
        assert data[0]["position"] == 0.0
        assert data[0]["emotion"] == "happy"
        assert data[0]["intensity"] == 0.8
        assert data[0]["easing"] == "ease_in"

    def test_from_keyframes_dict(self):
        """Test deserialization from dict."""
        data = [
            {"position": 0.0, "emotion": "happy", "intensity": 0.9},
            {"position": 1.0, "emotion": "sad"},
        ]
        curve = EmotionCurve.from_keyframes_dict(data)
        assert len(curve) == 2
        assert curve.keyframes[0].emotion == "happy"
        assert curve.keyframes[0].intensity == 0.9

    def test_len(self):
        """Test __len__ method."""
        curve = EmotionCurve()
        assert len(curve) == 0
        curve.add_point(0.0, "happy")
        assert len(curve) == 1

    def test_easing_functions_exist(self):
        """Test all easing functions exist."""
        expected = ["linear", "ease_in", "ease_out", "ease_in_out",
                    "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
                    "step", "hold"]
        for easing in expected:
            assert easing in EmotionCurve.EASINGS

    def test_easing_linear(self):
        """Test linear easing function."""
        fn = EmotionCurve.EASINGS["linear"]
        assert fn(0.0) == 0.0
        assert fn(0.5) == 0.5
        assert fn(1.0) == 1.0

    def test_easing_ease_in(self):
        """Test ease_in function (t^2)."""
        fn = EmotionCurve.EASINGS["ease_in"]
        assert fn(0.0) == 0.0
        assert fn(0.5) == 0.25
        assert fn(1.0) == 1.0

    def test_easing_step(self):
        """Test step easing function."""
        fn = EmotionCurve.EASINGS["step"]
        assert fn(0.0) == 0.0
        assert fn(0.49) == 0.0
        assert fn(0.5) == 1.0
        assert fn(1.0) == 1.0


# =============================================================================
# Curve Factory Function Tests
# =============================================================================

class TestCurveFactories:
    """Tests for curve factory functions."""

    def test_create_linear_curve(self):
        """Test creating linear curve."""
        curve = create_linear_curve("happy", "sad")
        assert len(curve) == 2
        assert curve.keyframes[0].emotion == "happy"
        assert curve.keyframes[1].emotion == "sad"

    def test_create_arc_curve(self):
        """Test creating arc curve."""
        curve = create_arc_curve("neutral", "excited", "happy")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "neutral"
        assert curve.keyframes[1].emotion == "excited"
        assert curve.keyframes[2].emotion == "happy"

    def test_create_arc_curve_custom_peak(self):
        """Test arc curve with custom peak position."""
        curve = create_arc_curve("neutral", "excited", "happy", peak_position=0.7)
        assert curve.keyframes[1].position == 0.7

    def test_create_buildup_curve(self):
        """Test creating buildup curve."""
        curve = create_buildup_curve("calm", "excited")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "calm"
        assert curve.keyframes[-1].emotion == "excited"

    def test_create_fade_curve(self):
        """Test creating fade curve."""
        curve = create_fade_curve("happy", "sad")
        assert len(curve) == 3
        assert curve.keyframes[0].emotion == "happy"
        assert curve.keyframes[-1].emotion == "sad"

    def test_create_wave_curve(self):
        """Test creating wave curve."""
        curve = create_wave_curve("calm", "excited", num_waves=2)
        assert len(curve) >= 3
        # Should alternate between emotions
        emotions = [kf.emotion for kf in curve.keyframes]
        assert "calm" in emotions
        assert "excited" in emotions


# =============================================================================
# NARRATIVE_CURVES Tests
# =============================================================================

class TestNarrativeCurves:
    """Tests for NARRATIVE_CURVES dictionary."""

    def test_contains_expected_curves(self):
        """Test expected curves exist."""
        expected = ["tension_build", "joy_arc", "sadness_fade", "suspense",
                    "revelation", "comfort", "anger_buildup", "resolution"]
        for name in expected:
            assert name in NARRATIVE_CURVES

    def test_curves_are_emotion_curves(self):
        """Test all values are EmotionCurve instances."""
        for name, curve in NARRATIVE_CURVES.items():
            assert isinstance(curve, EmotionCurve), f"{name} is not EmotionCurve"

    def test_get_narrative_curve_known(self):
        """Test getting known curve."""
        curve = get_narrative_curve("tension_build")
        assert isinstance(curve, EmotionCurve)

    def test_get_narrative_curve_unknown(self):
        """Test getting unknown curve returns None."""
        curve = get_narrative_curve("unknowncurvexyz")
        assert curve is None

    def test_get_narrative_curve_case_insensitive(self):
        """Test case insensitivity."""
        curve = get_narrative_curve("TENSION_BUILD")
        assert isinstance(curve, EmotionCurve)

    def test_list_narrative_curves(self):
        """Test listing all curves."""
        curves = list_narrative_curves()
        assert isinstance(curves, list)
        assert "tension_build" in curves
        assert "joy_arc" in curves


# =============================================================================
# ExtractorBackend Enum Tests
# =============================================================================

class TestExtractorBackend:
    """Tests for ExtractorBackend enum."""

    def test_resemblyzer_value(self):
        """Test RESEMBLYZER value."""
        assert ExtractorBackend.RESEMBLYZER.value == "resemblyzer"

    def test_speechbrain_value(self):
        """Test SPEECHBRAIN value."""
        assert ExtractorBackend.SPEECHBRAIN.value == "speechbrain"

    def test_wespeaker_value(self):
        """Test WESPEAKER value."""
        assert ExtractorBackend.WESPEAKER.value == "wespeaker"

    def test_mock_value(self):
        """Test MOCK value."""
        assert ExtractorBackend.MOCK.value == "mock"


# =============================================================================
# VoiceEmbedding Tests
# =============================================================================

class TestVoiceEmbedding:
    """Tests for VoiceEmbedding dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        emb = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(embedding=emb)
        assert ve.embedding_dim == 256
        assert ve.embedding_id != ""  # Auto-generated

    def test_embedding_id_generated(self):
        """Test embedding ID is auto-generated."""
        emb = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(embedding=emb)
        assert len(ve.embedding_id) == 16

    def test_embedding_id_deterministic(self):
        """Test same embedding produces same ID."""
        emb = np.ones(256, dtype=np.float32)
        ve1 = VoiceEmbedding(embedding=emb.copy())
        ve2 = VoiceEmbedding(embedding=emb.copy())
        assert ve1.embedding_id == ve2.embedding_id

    def test_similarity_identical(self):
        """Test similarity of identical embeddings."""
        emb = np.random.randn(256).astype(np.float32)
        ve1 = VoiceEmbedding(embedding=emb)
        ve2 = VoiceEmbedding(embedding=emb.copy())
        sim = ve1.similarity(ve2)
        assert abs(sim - 1.0) < 0.001

    def test_similarity_orthogonal(self):
        """Test similarity of orthogonal embeddings."""
        emb1 = np.zeros(256, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(256, dtype=np.float32)
        emb2[1] = 1.0
        ve1 = VoiceEmbedding(embedding=emb1)
        ve2 = VoiceEmbedding(embedding=emb2)
        sim = ve1.similarity(ve2)
        assert abs(sim) < 0.001

    def test_similarity_dimension_mismatch(self):
        """Test similarity with mismatched dimensions raises."""
        ve1 = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        ve2 = VoiceEmbedding(embedding=np.random.randn(128).astype(np.float32))
        with pytest.raises(ValueError, match="dimension mismatch"):
            ve1.similarity(ve2)

    def test_to_dict(self):
        """Test serialization to dict."""
        emb = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(
            embedding=emb,
            source_path="/test/audio.wav",
            quality_score=0.9,
        )
        data = ve.to_dict()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)
        assert data["source_path"] == "/test/audio.wav"
        assert data["quality_score"] == 0.9

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "embedding": [0.1] * 256,
            "embedding_dim": 256,
            "source_path": "/test.wav",
            "source_duration_seconds": 3.0,
            "source_sample_rate": 16000,
            "extractor_backend": "mock",
            "extraction_time": 0.1,
            "created_at": 1234567890.0,
            "quality_score": 0.8,
            "snr_db": 25.0,
            "estimated_gender": None,
            "estimated_age_range": None,
            "language_detected": None,
            "embedding_id": "abc123",
        }
        ve = VoiceEmbedding.from_dict(data)
        assert ve.source_path == "/test.wav"
        assert ve.quality_score == 0.8
        assert ve.embedding_id == "abc123"

    def test_save_and_load_json(self):
        """Test saving and loading JSON format."""
        emb = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(embedding=emb, source_path="/test.wav")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embedding.json"
            ve.save(path)
            loaded = VoiceEmbedding.load(path)
            assert loaded.source_path == "/test.wav"
            assert np.allclose(loaded.embedding, emb)

    def test_save_and_load_npz(self):
        """Test saving and loading NPZ format."""
        emb = np.random.randn(256).astype(np.float32)
        ve = VoiceEmbedding(embedding=emb, source_path="/test.wav")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "embedding.npz"
            ve.save(path)
            loaded = VoiceEmbedding.load(path)
            assert loaded.source_path == "/test.wav"
            assert np.allclose(loaded.embedding, emb)


# =============================================================================
# VoiceExtractor Tests
# =============================================================================

class TestVoiceExtractor:
    """Tests for VoiceExtractor class."""

    def test_init_default(self):
        """Test default initialization."""
        extractor = VoiceExtractor()
        assert extractor.backend == ExtractorBackend.MOCK
        assert extractor.device == "cpu"
        assert not extractor._loaded

    def test_init_custom(self):
        """Test custom initialization."""
        extractor = VoiceExtractor(
            backend=ExtractorBackend.MOCK,
            device="cuda",
        )
        assert extractor.device == "cuda"

    def test_embedding_dim_mock(self):
        """Test embedding dimension for mock backend."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        assert extractor.embedding_dim == 256

    def test_embedding_dim_speechbrain(self):
        """Test embedding dimension for speechbrain backend."""
        extractor = VoiceExtractor(backend=ExtractorBackend.SPEECHBRAIN)
        assert extractor.embedding_dim == 192

    def test_load_mock(self):
        """Test loading mock backend."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        extractor.load()
        assert extractor._loaded

    def test_unload(self):
        """Test unloading model."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        extractor.load()
        extractor.unload()
        assert not extractor._loaded
        assert extractor._model is None

    def test_extract_from_numpy(self):
        """Test extracting from numpy array."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.random.randn(16000).astype(np.float32)
        emb = extractor.extract(audio, sample_rate=16000)
        assert isinstance(emb, VoiceEmbedding)
        assert len(emb.embedding) == 256

    def test_extract_quality_score(self):
        """Test quality score estimation."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        emb = extractor.extract(audio, sample_rate=16000)
        assert 0.0 <= emb.quality_score <= 1.0

    def test_extract_from_segments(self):
        """Test extracting from segments."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        # 10 seconds of audio
        audio = np.random.randn(160000).astype(np.float32)
        embeddings = extractor.extract_from_segments(
            audio, segment_seconds=3.0, sample_rate=16000
        )
        assert len(embeddings) >= 1

    def test_average_embeddings(self):
        """Test averaging multiple embeddings."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio1 = np.random.randn(16000).astype(np.float32)
        audio2 = np.random.randn(16000).astype(np.float32)
        emb1 = extractor.extract(audio1)
        emb2 = extractor.extract(audio2)
        averaged = extractor.average_embeddings([emb1, emb2])
        assert isinstance(averaged, VoiceEmbedding)
        # Averaged should be normalized
        norm = np.linalg.norm(averaged.embedding)
        assert abs(norm - 1.0) < 0.01

    def test_average_embeddings_empty_raises(self):
        """Test averaging empty list raises."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        with pytest.raises(ValueError, match="No embeddings"):
            extractor.average_embeddings([])

    def test_preprocess_audio_normalize(self):
        """Test audio normalization."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        audio = np.array([0.0, 0.5, 1.0, -1.0], dtype=np.float32)
        processed = extractor._preprocess_audio(audio, 16000, 16000)
        assert np.abs(processed).max() <= 1.0

    def test_estimate_quality(self):
        """Test quality estimation."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        # Normal audio
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        quality, snr = extractor._estimate_quality(audio)
        assert 0.0 <= quality <= 1.0

    def test_estimate_quality_clipping_penalty(self):
        """Test clipping reduces quality."""
        extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)
        # Audio with heavy clipping
        audio = np.ones(16000, dtype=np.float32)
        quality, snr = extractor._estimate_quality(audio)
        assert quality < 1.0


# =============================================================================
# extract_embedding Convenience Function Tests
# =============================================================================

class TestExtractEmbeddingFunction:
    """Tests for extract_embedding convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        audio = np.random.randn(16000).astype(np.float32)
        emb = extract_embedding(audio)
        assert isinstance(emb, VoiceEmbedding)

    def test_with_string_backend(self):
        """Test with string backend."""
        audio = np.random.randn(16000).astype(np.float32)
        emb = extract_embedding(audio, backend="mock")
        assert isinstance(emb, VoiceEmbedding)


# =============================================================================
# EmotionStyle Enum Tests
# =============================================================================

class TestEmotionStyle:
    """Tests for EmotionStyle enum."""

    def test_neutral_value(self):
        """Test NEUTRAL value."""
        assert EmotionStyle.NEUTRAL.value == "neutral"

    def test_happy_value(self):
        """Test HAPPY value."""
        assert EmotionStyle.HAPPY.value == "happy"

    def test_all_emotions_exist(self):
        """Test all expected emotions exist."""
        expected = ["neutral", "happy", "sad", "angry", "fearful",
                    "surprised", "disgusted", "calm", "excited", "tender"]
        values = [e.value for e in EmotionStyle]
        for exp in expected:
            assert exp in values


# =============================================================================
# TimbreEmbedding Tests
# =============================================================================

class TestTimbreEmbedding:
    """Tests for TimbreEmbedding dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        emb = np.random.randn(256).astype(np.float32)
        te = TimbreEmbedding(embedding=emb)
        assert te.embedding_dim == 256
        assert te.separation_quality == 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        emb = np.random.randn(256).astype(np.float32)
        te = TimbreEmbedding(
            embedding=emb,
            source_voice_id="voice123",
            separation_quality=0.9,
        )
        data = te.to_dict()
        assert isinstance(data["embedding"], list)
        assert data["source_voice_id"] == "voice123"
        assert data["separation_quality"] == 0.9

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "embedding": [0.1] * 256,
            "embedding_dim": 256,
            "source_voice_id": "test",
            "source_embedding_id": None,
            "separation_quality": 0.8,
        }
        te = TimbreEmbedding.from_dict(data)
        assert te.source_voice_id == "test"
        assert te.separation_quality == 0.8


# =============================================================================
# EmotionEmbedding Tests
# =============================================================================

class TestEmotionEmbedding:
    """Tests for EmotionEmbedding dataclass."""

    def test_creation_basic(self):
        """Test basic creation."""
        emb = np.random.randn(64).astype(np.float32)
        ee = EmotionEmbedding(embedding=emb)
        assert ee.embedding_dim == 64
        assert ee.emotion_intensity == 1.0

    def test_creation_with_all_fields(self):
        """Test creation with all fields."""
        emb = np.random.randn(64).astype(np.float32)
        ee = EmotionEmbedding(
            embedding=emb,
            emotion_label="happy",
            emotion_intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
        )
        assert ee.emotion_label == "happy"
        assert ee.valence == 0.7

    def test_to_dict(self):
        """Test serialization."""
        emb = np.random.randn(64).astype(np.float32)
        ee = EmotionEmbedding(
            embedding=emb,
            emotion_label="sad",
            valence=-0.5,
        )
        data = ee.to_dict()
        assert data["emotion_label"] == "sad"
        assert data["valence"] == -0.5

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "embedding": [0.1] * 64,
            "embedding_dim": 64,
            "emotion_label": "angry",
            "emotion_intensity": 0.9,
            "valence": -0.6,
            "arousal": 0.8,
            "dominance": 0.7,
            "source_path": None,
        }
        ee = EmotionEmbedding.from_dict(data)
        assert ee.emotion_label == "angry"
        assert ee.arousal == 0.8


# =============================================================================
# SeparatedVoice Tests
# =============================================================================

class TestSeparatedVoice:
    """Tests for SeparatedVoice dataclass."""

    def test_creation(self):
        """Test basic creation."""
        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))
        sv = SeparatedVoice(timbre=timbre, emotion=emotion)
        assert sv.reconstruction_loss == 0.0

    def test_recombine(self):
        """Test recombining timbre and emotion."""
        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        emotion = EmotionEmbedding(embedding=np.random.randn(64).astype(np.float32))
        sv = SeparatedVoice(timbre=timbre, emotion=emotion)
        combined = sv.recombine()
        assert len(combined) == 256
        # Should be normalized
        norm = np.linalg.norm(combined)
        assert abs(norm - 1.0) < 0.01

    def test_with_emotion(self):
        """Test applying different emotion."""
        timbre = TimbreEmbedding(embedding=np.random.randn(256).astype(np.float32))
        original_emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="neutral"
        )
        new_emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="happy"
        )
        sv = SeparatedVoice(timbre=timbre, emotion=original_emotion)
        result = sv.with_emotion(new_emotion)
        assert len(result) == 256
        # Original emotion should be unchanged
        assert sv.emotion.emotion_label == "neutral"


# =============================================================================
# EmotionTimbreSeparator Tests
# =============================================================================

class TestEmotionTimbreSeparator:
    """Tests for EmotionTimbreSeparator class."""

    def test_init_default(self):
        """Test default initialization."""
        sep = EmotionTimbreSeparator()
        assert sep.timbre_dim == 256
        assert sep.emotion_dim == 64
        assert sep.device == "cpu"

    def test_init_custom(self):
        """Test custom initialization."""
        sep = EmotionTimbreSeparator(
            timbre_dim=512,
            emotion_dim=128,
            device="cuda",
        )
        assert sep.timbre_dim == 512
        assert sep.emotion_dim == 128

    def test_separate_from_numpy(self):
        """Test separating numpy array."""
        sep = EmotionTimbreSeparator()
        vector = np.random.randn(256).astype(np.float32)
        result = sep.separate(vector)
        assert isinstance(result, SeparatedVoice)
        assert result.timbre is not None
        assert result.emotion is not None

    def test_separate_from_voice_embedding(self):
        """Test separating VoiceEmbedding."""
        sep = EmotionTimbreSeparator()
        ve = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        result = sep.separate(ve)
        assert isinstance(result, SeparatedVoice)
        assert result.original_embedding is not None

    def test_transfer_emotion_with_preset(self):
        """Test transferring emotion preset."""
        sep = EmotionTimbreSeparator()
        ve = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        result = sep.transfer_emotion(ve, "happy")
        assert len(result) == 256

    def test_transfer_emotion_with_embedding(self):
        """Test transferring emotion embedding."""
        sep = EmotionTimbreSeparator()
        ve = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        emotion = EmotionEmbedding(
            embedding=np.random.randn(64).astype(np.float32),
            emotion_label="excited"
        )
        result = sep.transfer_emotion(ve, emotion)
        assert len(result) == 256

    def test_transfer_emotion_with_intensity(self):
        """Test transferring with intensity scaling."""
        sep = EmotionTimbreSeparator()
        ve = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        result = sep.transfer_emotion(ve, "happy", intensity=0.5)
        assert len(result) == 256

    def test_get_emotion_preset_known(self):
        """Test getting known emotion preset."""
        sep = EmotionTimbreSeparator()
        preset = sep.get_emotion_preset("happy")
        assert isinstance(preset, EmotionEmbedding)
        assert preset.emotion_label == "happy"
        assert preset.valence > 0

    def test_get_emotion_preset_unknown_raises(self):
        """Test getting unknown preset raises."""
        sep = EmotionTimbreSeparator()
        with pytest.raises(ValueError, match="Unknown emotion"):
            sep.get_emotion_preset("unknownemotionxyz")

    def test_list_emotion_presets(self):
        """Test listing emotion presets."""
        sep = EmotionTimbreSeparator()
        presets = sep.list_emotion_presets()
        assert "happy" in presets
        assert "sad" in presets
        assert "neutral" in presets

    def test_blend_emotions(self):
        """Test blending multiple emotions."""
        sep = EmotionTimbreSeparator()
        e1 = sep.get_emotion_preset("happy")
        e2 = sep.get_emotion_preset("sad")
        blended = sep.blend_emotions([(e1, 0.6), (e2, 0.4)])
        assert isinstance(blended, EmotionEmbedding)
        assert blended.emotion_label == "blended"

    def test_blend_emotions_empty(self):
        """Test blending empty list returns neutral."""
        sep = EmotionTimbreSeparator()
        result = sep.blend_emotions([])
        assert result.emotion_label == "neutral"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestSeparationConvenienceFunctions:
    """Tests for separation convenience functions."""

    def test_separate_voice(self):
        """Test separate_voice function."""
        vector = np.random.randn(256).astype(np.float32)
        result = separate_voice(vector)
        assert isinstance(result, SeparatedVoice)

    def test_transfer_emotion_function(self):
        """Test transfer_emotion function."""
        ve = VoiceEmbedding(embedding=np.random.randn(256).astype(np.float32))
        result = transfer_emotion(ve, "excited", intensity=0.8)
        assert len(result) == 256
