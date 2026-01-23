"""
Voice Cloner.

High-level API for voice cloning operations.
Combines extraction, library management, and synthesis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple

import numpy as np

from voice_soundboard.cloning.extractor import (
    VoiceEmbedding,
    VoiceExtractor,
    ExtractorBackend,
)
from voice_soundboard.cloning.library import (
    VoiceLibrary,
    VoiceProfile,
    get_default_library,
)


@dataclass
class CloningConfig:
    """Configuration for voice cloning operations."""

    # Extraction settings
    extractor_backend: ExtractorBackend = ExtractorBackend.MOCK
    device: str = "cpu"

    # Audio requirements
    min_audio_seconds: float = 1.0
    max_audio_seconds: float = 30.0
    optimal_audio_seconds: float = 5.0

    # Quality thresholds
    min_quality_score: float = 0.3
    min_snr_db: float = 10.0

    # Embedding processing
    use_segment_averaging: bool = True
    segment_length_seconds: float = 3.0

    # Security/consent
    require_consent: bool = True
    add_watermark: bool = False


@dataclass
class CloningResult:
    """Result of a voice cloning operation."""

    success: bool
    voice_id: str = ""
    profile: Optional[VoiceProfile] = None
    embedding: Optional[VoiceEmbedding] = None

    # Timing
    extraction_time: float = 0.0
    total_time: float = 0.0

    # Quality metrics
    quality_score: float = 0.0
    snr_db: float = 0.0
    audio_duration: float = 0.0

    # Errors/warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class VoiceCloner:
    """
    High-level interface for voice cloning.

    Handles:
    - Voice embedding extraction from audio
    - Quality validation and recommendations
    - Library management
    - Emotion-timbre separation (when supported)
    """

    def __init__(
        self,
        config: Optional[CloningConfig] = None,
        library: Optional[VoiceLibrary] = None,
    ):
        """
        Initialize the voice cloner.

        Args:
            config: Cloning configuration
            library: Voice library (uses default if not provided)
        """
        self.config = config or CloningConfig()
        self.library = library if library is not None else get_default_library()
        self._extractor: Optional[VoiceExtractor] = None

    @property
    def extractor(self) -> VoiceExtractor:
        """Get or create the voice extractor."""
        if self._extractor is None:
            self._extractor = VoiceExtractor(
                backend=self.config.extractor_backend,
                device=self.config.device,
            )
        return self._extractor

    def clone(
        self,
        audio: Union[Path, str, np.ndarray],
        voice_id: str,
        name: Optional[str] = None,
        sample_rate: int = 16000,
        consent_given: bool = False,
        consent_notes: str = "",
        tags: Optional[List[str]] = None,
        **profile_kwargs,
    ) -> CloningResult:
        """
        Clone a voice from audio.

        Args:
            audio: Audio file path or numpy array (3-10 seconds recommended)
            voice_id: Unique identifier for the cloned voice
            name: Display name (defaults to voice_id)
            sample_rate: Sample rate if audio is numpy array
            consent_given: Whether consent for cloning was given
            consent_notes: Notes about consent
            tags: Tags for the voice profile
            **profile_kwargs: Additional VoiceProfile fields

        Returns:
            CloningResult with status and voice profile
        """
        start_time = time.time()
        warnings: List[str] = []
        recommendations: List[str] = []

        # Check consent if required
        if self.config.require_consent and not consent_given:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error="Consent is required for voice cloning. "
                "Set consent_given=True to acknowledge consent.",
            )

        # Check if voice already exists
        if voice_id in self.library:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error=f"Voice '{voice_id}' already exists. "
                "Use update_voice() to modify or delete first.",
            )

        # Validate audio
        source_path = None
        if isinstance(audio, (str, Path)):
            source_path = str(audio)
            path = Path(audio)
            if not path.exists():
                return CloningResult(
                    success=False,
                    voice_id=voice_id,
                    error=f"Audio file not found: {audio}",
                )

        # Extract embedding
        try:
            if self.config.use_segment_averaging and isinstance(audio, (str, Path)):
                # Extract from multiple segments and average
                embeddings = self.extractor.extract_from_segments(
                    audio,
                    segment_seconds=self.config.segment_length_seconds,
                    sample_rate=sample_rate,
                )
                if not embeddings:
                    return CloningResult(
                        success=False,
                        voice_id=voice_id,
                        error="Audio too short for segment extraction",
                    )
                embedding = self.extractor.average_embeddings(embeddings)
            else:
                embedding = self.extractor.extract(audio, sample_rate)

        except Exception as e:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error=f"Extraction failed: {str(e)}",
            )

        # Check quality
        if embedding.quality_score < self.config.min_quality_score:
            warnings.append(
                f"Low audio quality ({embedding.quality_score:.2f}). "
                "Results may be suboptimal."
            )
            recommendations.append("Use cleaner audio with less background noise")

        if embedding.snr_db < self.config.min_snr_db:
            warnings.append(
                f"Low signal-to-noise ratio ({embedding.snr_db:.1f} dB). "
            )
            recommendations.append("Record in a quieter environment")

        # Check duration
        if embedding.source_duration_seconds < self.config.min_audio_seconds:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error=f"Audio too short ({embedding.source_duration_seconds:.1f}s). "
                f"Minimum {self.config.min_audio_seconds}s required.",
            )

        if embedding.source_duration_seconds < self.config.optimal_audio_seconds:
            recommendations.append(
                f"For best results, use {self.config.optimal_audio_seconds}+ seconds of audio"
            )

        if embedding.source_duration_seconds > self.config.max_audio_seconds:
            warnings.append(
                f"Audio longer than optimal ({embedding.source_duration_seconds:.1f}s). "
                "Only first portion used."
            )

        # Add to library
        try:
            profile = self.library.add(
                voice_id=voice_id,
                name=name or voice_id,
                embedding=embedding,
                source_audio_path=source_path,
                consent_given=consent_given,
                consent_notes=consent_notes,
                consent_date=time.strftime("%Y-%m-%d"),
                tags=tags or [],
                **profile_kwargs,
            )
        except Exception as e:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                embedding=embedding,
                error=f"Failed to save to library: {str(e)}",
            )

        total_time = time.time() - start_time

        return CloningResult(
            success=True,
            voice_id=voice_id,
            profile=profile,
            embedding=embedding,
            extraction_time=embedding.extraction_time,
            total_time=total_time,
            quality_score=embedding.quality_score,
            snr_db=embedding.snr_db,
            audio_duration=embedding.source_duration_seconds,
            warnings=warnings,
            recommendations=recommendations,
        )

    def clone_quick(
        self,
        audio: Union[Path, str, np.ndarray],
        name: str,
        sample_rate: int = 16000,
    ) -> CloningResult:
        """
        Quick clone without saving to library.

        Useful for one-off cloning or testing.

        Args:
            audio: Audio source
            name: Name for the voice
            sample_rate: Sample rate if numpy array

        Returns:
            CloningResult with embedding but no saved profile
        """
        start_time = time.time()

        try:
            embedding = self.extractor.extract(audio, sample_rate)
        except Exception as e:
            return CloningResult(
                success=False,
                error=f"Extraction failed: {str(e)}",
            )

        # Create temporary profile (not saved)
        profile = VoiceProfile(
            voice_id=f"temp_{int(time.time())}",
            name=name,
            embedding=embedding,
            source_duration_seconds=embedding.source_duration_seconds,
        )

        return CloningResult(
            success=True,
            voice_id=profile.voice_id,
            profile=profile,
            embedding=embedding,
            extraction_time=embedding.extraction_time,
            total_time=time.time() - start_time,
            quality_score=embedding.quality_score,
            snr_db=embedding.snr_db,
            audio_duration=embedding.source_duration_seconds,
        )

    def update_voice(
        self,
        voice_id: str,
        audio: Optional[Union[Path, str, np.ndarray]] = None,
        sample_rate: int = 16000,
        **updates,
    ) -> CloningResult:
        """
        Update an existing voice profile.

        Args:
            voice_id: Voice to update
            audio: New audio to extract embedding from (optional)
            sample_rate: Sample rate if numpy array
            **updates: Profile fields to update

        Returns:
            CloningResult with updated profile
        """
        start_time = time.time()

        try:
            profile = self.library.get_or_raise(voice_id)
        except KeyError as e:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error=str(e),
            )

        embedding = profile.embedding

        # Re-extract if new audio provided
        if audio is not None:
            try:
                embedding = self.extractor.extract(audio, sample_rate)
                updates["embedding"] = embedding
                updates["source_duration_seconds"] = embedding.source_duration_seconds
            except Exception as e:
                return CloningResult(
                    success=False,
                    voice_id=voice_id,
                    error=f"Re-extraction failed: {str(e)}",
                )

        # Apply updates
        try:
            profile = self.library.update(voice_id, **updates)
        except Exception as e:
            return CloningResult(
                success=False,
                voice_id=voice_id,
                error=f"Update failed: {str(e)}",
            )

        return CloningResult(
            success=True,
            voice_id=voice_id,
            profile=profile,
            embedding=embedding,
            total_time=time.time() - start_time,
            quality_score=embedding.quality_score if embedding else 0,
            snr_db=embedding.snr_db if embedding else 0,
        )

    def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a voice from the library.

        Args:
            voice_id: Voice to delete

        Returns:
            True if deleted, False if not found
        """
        return self.library.remove(voice_id)

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """
        Get a voice profile.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceProfile or None
        """
        return self.library.get(voice_id)

    def list_voices(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **filters,
    ) -> List[VoiceProfile]:
        """
        List/search voice profiles.

        Args:
            query: Text search
            tags: Required tags
            **filters: Additional filters (gender, language, etc.)

        Returns:
            Matching voice profiles
        """
        return self.library.search(query=query, tags=tags, **filters)

    def find_similar(
        self,
        audio: Union[Path, str, np.ndarray, VoiceEmbedding],
        top_k: int = 5,
        sample_rate: int = 16000,
    ) -> List[Tuple[VoiceProfile, float]]:
        """
        Find similar voices in the library.

        Args:
            audio: Audio or embedding to compare
            top_k: Maximum results
            sample_rate: Sample rate if numpy array

        Returns:
            List of (VoiceProfile, similarity) tuples
        """
        if isinstance(audio, VoiceEmbedding):
            embedding = audio
        else:
            embedding = self.extractor.extract(audio, sample_rate)

        return self.library.find_similar(embedding, top_k=top_k)

    def validate_audio(
        self,
        audio: Union[Path, str, np.ndarray],
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Validate audio for voice cloning without cloning.

        Args:
            audio: Audio to validate
            sample_rate: Sample rate if numpy array

        Returns:
            Validation results with recommendations
        """
        issues = []
        recommendations = []
        is_valid = True

        try:
            embedding = self.extractor.extract(audio, sample_rate)
        except Exception as e:
            return {
                "is_valid": False,
                "issues": [f"Cannot process audio: {str(e)}"],
                "recommendations": ["Ensure audio is valid WAV/MP3"],
            }

        # Check duration
        if embedding.source_duration_seconds < self.config.min_audio_seconds:
            is_valid = False
            issues.append(
                f"Too short: {embedding.source_duration_seconds:.1f}s "
                f"(need {self.config.min_audio_seconds}s+)"
            )
        elif embedding.source_duration_seconds < self.config.optimal_audio_seconds:
            recommendations.append(
                f"Consider using {self.config.optimal_audio_seconds}+ seconds"
            )

        # Check quality
        if embedding.quality_score < self.config.min_quality_score:
            is_valid = False
            issues.append(f"Quality too low: {embedding.quality_score:.2f}")
            recommendations.append("Use cleaner audio with less noise/clipping")
        elif embedding.quality_score < 0.7:
            recommendations.append("Audio quality is acceptable but could be improved")

        # Check SNR
        if embedding.snr_db < self.config.min_snr_db:
            issues.append(f"SNR too low: {embedding.snr_db:.1f} dB")
            recommendations.append("Record in a quieter environment")

        return {
            "is_valid": is_valid,
            "duration_seconds": embedding.source_duration_seconds,
            "quality_score": embedding.quality_score,
            "snr_db": embedding.snr_db,
            "issues": issues,
            "recommendations": recommendations,
        }

    def export_voice(
        self,
        voice_id: str,
        output_path: Union[Path, str],
        include_source_audio: bool = False,
    ) -> Path:
        """
        Export a voice profile for sharing/backup.

        Args:
            voice_id: Voice to export
            output_path: Output file path (.json or .npz)
            include_source_audio: Whether to include source audio

        Returns:
            Path to exported file
        """
        profile = self.library.get_or_raise(voice_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = profile.to_dict()
        if not include_source_audio:
            data["source_audio_path"] = None

        if output_path.suffix == ".npz":
            import json
            import numpy as np

            np.savez_compressed(
                output_path,
                embedding=profile.embedding.embedding if profile.embedding else [],
                metadata=json.dumps({k: v for k, v in data.items() if k != "embedding"}),
            )
        else:
            import json

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        return output_path

    def import_voice(
        self,
        input_path: Union[Path, str],
        voice_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> VoiceProfile:
        """
        Import a voice profile from file.

        Args:
            input_path: Path to exported voice file
            voice_id: Override voice ID (optional)
            overwrite: Whether to overwrite existing voice

        Returns:
            Imported VoiceProfile
        """
        input_path = Path(input_path)

        if input_path.suffix == ".npz":
            import json

            data = np.load(input_path, allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))
            metadata["embedding"] = {
                "embedding": data["embedding"].tolist(),
                "embedding_dim": len(data["embedding"]),
            }
            profile_data = metadata
        else:
            import json

            with open(input_path) as f:
                profile_data = json.load(f)

        # Override voice_id if provided
        if voice_id:
            profile_data["voice_id"] = voice_id

        actual_id = profile_data["voice_id"]

        # Check for existing
        if actual_id in self.library:
            if overwrite:
                self.library.remove(actual_id)
            else:
                raise ValueError(
                    f"Voice '{actual_id}' already exists. Set overwrite=True to replace."
                )

        # Reconstruct profile and add
        profile = VoiceProfile.from_dict(profile_data)

        self.library.add(
            voice_id=profile.voice_id,
            name=profile.name,
            embedding=profile.embedding,
            description=profile.description,
            tags=profile.tags,
            gender=profile.gender,
            language=profile.language,
            consent_given=profile.consent_given,
            consent_notes=profile.consent_notes,
        )

        return profile
