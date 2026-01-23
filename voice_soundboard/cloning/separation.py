"""
Emotion-Timbre Separation.

Disentangles speaker timbre (voice identity) from emotional expression,
enabling emotion transfer between speakers.

Based on research from:
- IndexTTS2: Emotion-timbre disentanglement
- Marco-Voice: Alibaba's emotion-timbre separation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple, Union
from pathlib import Path
from enum import Enum

import numpy as np

from voice_soundboard.cloning.extractor import VoiceEmbedding, VoiceExtractor


class EmotionStyle(Enum):
    """Standard emotion styles for transfer."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CALM = "calm"
    EXCITED = "excited"
    TENDER = "tender"


@dataclass
class TimbreEmbedding:
    """
    Speaker timbre embedding (identity without emotion).

    Represents the speaker's unique vocal characteristics
    independent of emotional expression.
    """

    embedding: np.ndarray
    embedding_dim: int = 256

    # Source info
    source_voice_id: Optional[str] = None
    source_embedding_id: Optional[str] = None

    # Quality
    separation_quality: float = 1.0  # How clean the separation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "embedding": self.embedding.tolist(),
            "embedding_dim": self.embedding_dim,
            "source_voice_id": self.source_voice_id,
            "source_embedding_id": self.source_embedding_id,
            "separation_quality": self.separation_quality,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimbreEmbedding":
        """Deserialize from dictionary."""
        data = data.copy()
        data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return cls(**data)


@dataclass
class EmotionEmbedding:
    """
    Emotion expression embedding (style without identity).

    Represents the emotional expression characteristics
    that can be transferred to other speakers.
    """

    embedding: np.ndarray
    embedding_dim: int = 64  # Typically smaller than timbre

    # Emotion info
    emotion_label: Optional[str] = None
    emotion_intensity: float = 1.0  # 0-1

    # VAD values (if available)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None

    # Source info
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "embedding": self.embedding.tolist(),
            "embedding_dim": self.embedding_dim,
            "emotion_label": self.emotion_label,
            "emotion_intensity": self.emotion_intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionEmbedding":
        """Deserialize from dictionary."""
        data = data.copy()
        data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return cls(**data)


@dataclass
class SeparatedVoice:
    """
    A voice separated into timbre and emotion components.
    """

    timbre: TimbreEmbedding
    emotion: EmotionEmbedding

    # Original embedding for reference
    original_embedding: Optional[VoiceEmbedding] = None

    # Reconstruction quality
    reconstruction_loss: float = 0.0

    def recombine(self) -> np.ndarray:
        """
        Recombine timbre and emotion into a voice embedding.

        Returns:
            Combined embedding vector
        """
        # Simple concatenation + projection (mock implementation)
        # Real implementation would use a learned decoder
        combined = np.concatenate([
            self.timbre.embedding * 0.7,
            np.tile(self.emotion.embedding, 4)[:256] * 0.3,
        ])[:256]

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def with_emotion(self, emotion: EmotionEmbedding) -> np.ndarray:
        """
        Recombine with a different emotion.

        Args:
            emotion: New emotion embedding to apply

        Returns:
            Combined embedding with new emotion
        """
        original_emotion = self.emotion
        self.emotion = emotion
        result = self.recombine()
        self.emotion = original_emotion
        return result


class EmotionTimbreSeparator:
    """
    Separates voice embeddings into timbre and emotion components.

    This enables:
    - Applying one speaker's emotion to another's voice
    - Extracting "emotion templates" from expressive speech
    - Creating emotionally neutral voice profiles
    """

    def __init__(
        self,
        timbre_dim: int = 256,
        emotion_dim: int = 64,
        device: str = "cpu",
    ):
        """
        Initialize the separator.

        Args:
            timbre_dim: Dimension of timbre embeddings
            emotion_dim: Dimension of emotion embeddings
            device: Device for computation
        """
        self.timbre_dim = timbre_dim
        self.emotion_dim = emotion_dim
        self.device = device
        self._model = None

    def separate(
        self,
        embedding: Union[VoiceEmbedding, np.ndarray],
    ) -> SeparatedVoice:
        """
        Separate a voice embedding into timbre and emotion.

        Args:
            embedding: Voice embedding to separate

        Returns:
            SeparatedVoice with timbre and emotion components
        """
        if isinstance(embedding, VoiceEmbedding):
            vector = embedding.embedding
            source_id = embedding.embedding_id
        else:
            vector = embedding
            source_id = None

        # Mock separation (in production, use a trained model)
        # The idea is that timbre is the "average" voice characteristics
        # while emotion is the deviation from that average

        # Simulate timbre extraction (low-frequency components)
        timbre_vector = self._extract_timbre(vector)

        # Simulate emotion extraction (deviation from neutral)
        emotion_vector = self._extract_emotion(vector, timbre_vector)

        timbre = TimbreEmbedding(
            embedding=timbre_vector,
            embedding_dim=len(timbre_vector),
            source_embedding_id=source_id,
        )

        emotion = EmotionEmbedding(
            embedding=emotion_vector,
            embedding_dim=len(emotion_vector),
        )

        return SeparatedVoice(
            timbre=timbre,
            emotion=emotion,
            original_embedding=embedding if isinstance(embedding, VoiceEmbedding) else None,
        )

    def _extract_timbre(self, vector: np.ndarray) -> np.ndarray:
        """Extract timbre component (speaker identity)."""
        # Simplified: take "stable" components
        # In reality, this would be a learned projection

        # Smooth the vector (simulating low-frequency extraction)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(vector, kernel, mode="same")

        # Normalize
        norm = np.linalg.norm(smoothed)
        if norm > 0:
            smoothed = smoothed / norm

        return smoothed.astype(np.float32)

    def _extract_emotion(
        self,
        vector: np.ndarray,
        timbre: np.ndarray,
    ) -> np.ndarray:
        """Extract emotion component (expression style)."""
        # Simplified: emotion is the residual after removing timbre
        residual = vector - timbre * 0.8

        # Project to smaller dimension
        # In reality, use a learned projection
        indices = np.linspace(0, len(residual) - 1, self.emotion_dim).astype(int)
        emotion = residual[indices]

        # Normalize
        norm = np.linalg.norm(emotion)
        if norm > 0:
            emotion = emotion / norm

        return emotion.astype(np.float32)

    def transfer_emotion(
        self,
        source_voice: Union[VoiceEmbedding, SeparatedVoice],
        emotion_reference: Union[VoiceEmbedding, EmotionEmbedding, str],
        intensity: float = 1.0,
    ) -> np.ndarray:
        """
        Transfer emotion from one voice to another.

        Args:
            source_voice: Voice to apply emotion to
            emotion_reference: Source of emotion (embedding, audio path, or emotion name)
            intensity: Emotion intensity multiplier

        Returns:
            Combined embedding with transferred emotion
        """
        # Get separated source voice
        if isinstance(source_voice, SeparatedVoice):
            separated = source_voice
        else:
            separated = self.separate(source_voice)

        # Get emotion embedding
        if isinstance(emotion_reference, EmotionEmbedding):
            emotion = emotion_reference
        elif isinstance(emotion_reference, VoiceEmbedding):
            emotion = self.separate(emotion_reference).emotion
        elif isinstance(emotion_reference, str):
            # Standard emotion preset
            emotion = self.get_emotion_preset(emotion_reference)
        else:
            raise ValueError(f"Unknown emotion reference type: {type(emotion_reference)}")

        # Apply intensity scaling
        if intensity != 1.0:
            scaled_embedding = emotion.embedding * intensity
            emotion = EmotionEmbedding(
                embedding=scaled_embedding,
                embedding_dim=len(scaled_embedding),
                emotion_label=emotion.emotion_label,
                emotion_intensity=intensity,
            )

        return separated.with_emotion(emotion)

    def get_emotion_preset(self, emotion_name: str) -> EmotionEmbedding:
        """
        Get a preset emotion embedding.

        Args:
            emotion_name: Name of emotion (e.g., "happy", "sad")

        Returns:
            EmotionEmbedding for the emotion
        """
        # Map emotion to VAD values
        emotion_vad = {
            "neutral": (0.0, 0.3, 0.5),
            "happy": (0.8, 0.6, 0.6),
            "sad": (-0.6, 0.3, 0.3),
            "angry": (-0.5, 0.8, 0.8),
            "fearful": (-0.7, 0.7, 0.2),
            "surprised": (0.3, 0.8, 0.4),
            "disgusted": (-0.6, 0.5, 0.6),
            "calm": (0.3, 0.2, 0.5),
            "excited": (0.7, 0.9, 0.7),
            "tender": (0.6, 0.3, 0.4),
        }

        emotion_lower = emotion_name.lower()
        if emotion_lower not in emotion_vad:
            raise ValueError(f"Unknown emotion: {emotion_name}")

        valence, arousal, dominance = emotion_vad[emotion_lower]

        # Create synthetic emotion embedding from VAD
        np.random.seed(hash(emotion_lower) % (2**31))
        base = np.random.randn(self.emotion_dim).astype(np.float32)

        # Modulate by VAD
        base[:self.emotion_dim // 3] *= valence
        base[self.emotion_dim // 3 : 2 * self.emotion_dim // 3] *= arousal
        base[2 * self.emotion_dim // 3 :] *= dominance

        # Normalize
        norm = np.linalg.norm(base)
        if norm > 0:
            base = base / norm

        return EmotionEmbedding(
            embedding=base,
            embedding_dim=self.emotion_dim,
            emotion_label=emotion_lower,
            emotion_intensity=1.0,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

    def list_emotion_presets(self) -> List[str]:
        """List available emotion presets."""
        return [e.value for e in EmotionStyle]

    def extract_emotion_from_audio(
        self,
        audio_path: Union[str, Path],
        extractor: Optional[VoiceExtractor] = None,
    ) -> EmotionEmbedding:
        """
        Extract emotion embedding from audio.

        Args:
            audio_path: Path to audio file
            extractor: Optional voice extractor

        Returns:
            Extracted emotion embedding
        """
        from voice_soundboard.cloning.extractor import (
            VoiceExtractor,
            ExtractorBackend,
        )

        if extractor is None:
            extractor = VoiceExtractor(backend=ExtractorBackend.MOCK)

        # Extract full embedding
        full_embedding = extractor.extract(audio_path)

        # Separate to get emotion component
        separated = self.separate(full_embedding)
        separated.emotion.source_path = str(audio_path)

        return separated.emotion

    def blend_emotions(
        self,
        emotions: List[Tuple[EmotionEmbedding, float]],
    ) -> EmotionEmbedding:
        """
        Blend multiple emotion embeddings.

        Args:
            emotions: List of (emotion, weight) tuples

        Returns:
            Blended emotion embedding
        """
        if not emotions:
            return self.get_emotion_preset("neutral")

        # Weighted average
        total_weight = sum(w for _, w in emotions)
        if total_weight == 0:
            total_weight = 1

        blended = np.zeros(self.emotion_dim, dtype=np.float32)
        blended_valence = 0.0
        blended_arousal = 0.0
        blended_dominance = 0.0

        for emotion, weight in emotions:
            normalized_weight = weight / total_weight
            blended += emotion.embedding * normalized_weight

            if emotion.valence is not None:
                blended_valence += emotion.valence * normalized_weight
            if emotion.arousal is not None:
                blended_arousal += emotion.arousal * normalized_weight
            if emotion.dominance is not None:
                blended_dominance += emotion.dominance * normalized_weight

        # Normalize
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return EmotionEmbedding(
            embedding=blended,
            embedding_dim=self.emotion_dim,
            emotion_label="blended",
            valence=blended_valence if blended_valence != 0 else None,
            arousal=blended_arousal if blended_arousal != 0 else None,
            dominance=blended_dominance if blended_dominance != 0 else None,
        )


def separate_voice(
    embedding: Union[VoiceEmbedding, np.ndarray],
) -> SeparatedVoice:
    """
    Convenience function to separate a voice embedding.

    Args:
        embedding: Voice embedding to separate

    Returns:
        SeparatedVoice with timbre and emotion components
    """
    separator = EmotionTimbreSeparator()
    return separator.separate(embedding)


def transfer_emotion(
    voice: VoiceEmbedding,
    emotion: str,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Convenience function to transfer emotion to a voice.

    Args:
        voice: Target voice embedding
        emotion: Emotion name to apply
        intensity: Emotion intensity

    Returns:
        Combined embedding with emotion applied
    """
    separator = EmotionTimbreSeparator()
    return separator.transfer_emotion(voice, emotion, intensity)
