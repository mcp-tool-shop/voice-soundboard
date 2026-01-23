"""
Voice Library Management.

Stores and manages cloned voice profiles for reuse.
Supports tagging, search, and organization.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any, Iterator
from datetime import datetime

from voice_soundboard.cloning.extractor import VoiceEmbedding

logger = logging.getLogger(__name__)


# Default library location
DEFAULT_LIBRARY_PATH = Path.home() / ".voice_soundboard" / "voices"


@dataclass
class VoiceProfile:
    """
    A saved voice profile with metadata.

    Contains the voice embedding plus additional info
    for organization and synthesis parameters.
    """

    # Identity
    voice_id: str
    name: str
    description: str = ""

    # Voice embedding
    embedding: Optional[VoiceEmbedding] = None

    # Source info
    source_audio_path: Optional[str] = None
    source_duration_seconds: float = 0.0

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    # Speaker characteristics
    gender: Optional[str] = None  # "male", "female", "neutral"
    age_range: Optional[str] = None  # "child", "young", "adult", "senior"
    accent: Optional[str] = None  # "american", "british", "australian", etc.
    language: str = "en"

    # Synthesis defaults
    default_speed: float = 1.0
    default_emotion: Optional[str] = None

    # Quality & usage
    quality_rating: float = 1.0  # User rating 0-1
    usage_count: int = 0
    last_used_at: Optional[float] = None

    # Consent tracking (important for voice cloning ethics)
    consent_given: bool = False
    consent_date: Optional[str] = None
    consent_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding.to_dict()

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            return obj

        return convert_numpy(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        """Deserialize profile from dictionary."""
        data = data.copy()
        if data.get("embedding"):
            data["embedding"] = VoiceEmbedding.from_dict(data["embedding"])
        return cls(**data)

    @property
    def created_date(self) -> str:
        """Get formatted creation date."""
        return datetime.fromtimestamp(self.created_at).strftime("%Y-%m-%d %H:%M")

    def record_usage(self) -> None:
        """Record that this voice was used."""
        self.usage_count += 1
        self.last_used_at = time.time()


class VoiceLibrary:
    """
    Manages a collection of cloned voice profiles.

    Provides CRUD operations, search, and organization.
    """

    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize the voice library.

        Args:
            library_path: Directory to store voice profiles
        """
        self.library_path = Path(library_path) if library_path else DEFAULT_LIBRARY_PATH
        self._profiles: Dict[str, VoiceProfile] = {}
        self._index_path = self.library_path / "index.json"
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ensure library is loaded from disk."""
        if not self._loaded:
            self.load()

    def load(self) -> None:
        """
        Load library index and profiles from disk.

        Raises:
            RuntimeError: If library cannot be created or loaded
        """
        try:
            self.library_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create voice library directory at {self.library_path}: {e}"
            ) from e

        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    index = json.load(f)
            except json.JSONDecodeError as e:
                logger.error("Corrupted library index at %s: %s", self._index_path, e)
                # Start with empty library rather than failing
                logger.warning("Starting with empty voice library due to corrupted index")
                index = {"voices": []}
            except OSError as e:
                raise RuntimeError(
                    f"Failed to read library index at {self._index_path}: {e}"
                ) from e

            for voice_id in index.get("voices", []):
                profile_path = self.library_path / voice_id / "profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path) as f:
                            self._profiles[voice_id] = VoiceProfile.from_dict(json.load(f))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping corrupted voice profile %s: %s", voice_id, e
                        )
                    except OSError as e:
                        logger.warning(
                            "Failed to read voice profile %s: %s", voice_id, e
                        )

        self._loaded = True

    def save(self) -> None:
        """
        Save library index to disk.

        Raises:
            RuntimeError: If index cannot be saved
        """
        try:
            self.library_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create voice library directory at {self.library_path}: {e}"
            ) from e

        # Save index
        index = {
            "version": "1.0",
            "voices": list(self._profiles.keys()),
            "updated_at": time.time(),
        }
        try:
            with open(self._index_path, "w") as f:
                json.dump(index, f, indent=2)
        except OSError as e:
            raise RuntimeError(
                f"Failed to save library index at {self._index_path}: {e}"
            ) from e

    def add(
        self,
        voice_id: str,
        name: str,
        embedding: VoiceEmbedding,
        source_audio_path: Optional[str] = None,
        copy_source: bool = True,
        **kwargs,
    ) -> VoiceProfile:
        """
        Add a new voice profile to the library.

        Args:
            voice_id: Unique identifier for the voice
            name: Display name
            embedding: Voice embedding from extraction
            source_audio_path: Path to source audio (will be copied if copy_source=True)
            copy_source: Whether to copy source audio to library
            **kwargs: Additional VoiceProfile fields

        Returns:
            Created VoiceProfile

        Raises:
            ValueError: If voice_id already exists
        """
        self._ensure_loaded()

        if voice_id in self._profiles:
            raise ValueError(f"Voice '{voice_id}' already exists in library")

        # Create profile directory
        profile_dir = self.library_path / voice_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        # Copy source audio if provided
        stored_audio_path = None
        if source_audio_path and copy_source:
            source = Path(source_audio_path)
            if source.exists():
                dest = profile_dir / f"source{source.suffix}"
                shutil.copy2(source, dest)
                stored_audio_path = str(dest)

        # Save embedding
        embedding_path = profile_dir / "embedding.npz"
        embedding.save(embedding_path)

        # Create profile
        profile = VoiceProfile(
            voice_id=voice_id,
            name=name,
            embedding=embedding,
            source_audio_path=stored_audio_path or source_audio_path,
            source_duration_seconds=embedding.source_duration_seconds,
            **kwargs,
        )

        # Save profile
        with open(profile_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

        self._profiles[voice_id] = profile
        self.save()

        return profile

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        """
        Get a voice profile by ID.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceProfile or None if not found
        """
        self._ensure_loaded()
        return self._profiles.get(voice_id)

    def get_or_raise(self, voice_id: str) -> VoiceProfile:
        """
        Get a voice profile or raise if not found.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceProfile

        Raises:
            KeyError: If voice not found
        """
        profile = self.get(voice_id)
        if profile is None:
            raise KeyError(f"Voice '{voice_id}' not found in library")
        return profile

    def update(self, voice_id: str, **updates) -> VoiceProfile:
        """
        Update a voice profile.

        Args:
            voice_id: Voice to update
            **updates: Fields to update

        Returns:
            Updated VoiceProfile

        Raises:
            KeyError: If voice not found
        """
        profile = self.get_or_raise(voice_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.updated_at = time.time()

        # Save updated profile
        profile_dir = self.library_path / voice_id
        with open(profile_dir / "profile.json", "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

        return profile

    def remove(self, voice_id: str, delete_files: bool = True) -> bool:
        """
        Remove a voice from the library.

        Args:
            voice_id: Voice to remove
            delete_files: Whether to delete files from disk

        Returns:
            True if removed, False if not found
        """
        self._ensure_loaded()

        if voice_id not in self._profiles:
            return False

        if delete_files:
            profile_dir = self.library_path / voice_id
            if profile_dir.exists():
                shutil.rmtree(profile_dir)

        del self._profiles[voice_id]
        self.save()
        return True

    def list_all(self) -> List[VoiceProfile]:
        """Get all voice profiles."""
        self._ensure_loaded()
        return list(self._profiles.values())

    def list_ids(self) -> List[str]:
        """Get all voice IDs."""
        self._ensure_loaded()
        return list(self._profiles.keys())

    def search(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        gender: Optional[str] = None,
        language: Optional[str] = None,
        min_quality: float = 0.0,
    ) -> List[VoiceProfile]:
        """
        Search voice profiles by criteria.

        Args:
            query: Text search in name/description
            tags: Required tags (any match)
            gender: Filter by gender
            language: Filter by language
            min_quality: Minimum quality rating

        Returns:
            Matching VoiceProfiles
        """
        self._ensure_loaded()
        results = []

        for profile in self._profiles.values():
            # Text search
            if query:
                query_lower = query.lower()
                if (
                    query_lower not in profile.name.lower()
                    and query_lower not in profile.description.lower()
                ):
                    continue

            # Tag filter
            if tags:
                if not any(tag in profile.tags for tag in tags):
                    continue

            # Gender filter
            if gender and profile.gender != gender:
                continue

            # Language filter
            if language and profile.language != language:
                continue

            # Quality filter
            if profile.quality_rating < min_quality:
                continue

            results.append(profile)

        return results

    def find_similar(
        self,
        embedding: VoiceEmbedding,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[tuple[VoiceProfile, float]]:
        """
        Find voices similar to given embedding.

        Args:
            embedding: Reference embedding
            top_k: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (VoiceProfile, similarity_score) tuples
        """
        self._ensure_loaded()
        results = []

        for profile in self._profiles.values():
            if profile.embedding is None:
                continue

            similarity = embedding.similarity(profile.embedding)
            if similarity >= min_similarity:
                results.append((profile, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        """Number of voices in library."""
        self._ensure_loaded()
        return len(self._profiles)

    def __contains__(self, voice_id: str) -> bool:
        """Check if voice exists in library."""
        self._ensure_loaded()
        return voice_id in self._profiles

    def __iter__(self) -> Iterator[VoiceProfile]:
        """Iterate over voice profiles."""
        self._ensure_loaded()
        return iter(self._profiles.values())


# Global default library instance
_default_library: Optional[VoiceLibrary] = None


def get_default_library() -> VoiceLibrary:
    """Get or create the default voice library."""
    global _default_library
    if _default_library is None:
        _default_library = VoiceLibrary()
    return _default_library
