"""
Voice Preset Catalog

Central catalog for managing voice presets from multiple sources.
Provides loading, filtering, search, and application of presets.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field

from .schema import (
    VoicePreset,
    PresetCatalogMetadata,
    PresetSource,
    AcousticParams,
    Gender,
    AgeRange,
    VoiceEnergy,
    VoiceTone,
)

logger = logging.getLogger(__name__)


# Path to catalog JSON files
CATALOG_DIR = Path(__file__).parent / "catalog"


@dataclass
class SearchResult:
    """A search result with relevance score."""
    preset: VoicePreset
    score: float = 1.0
    match_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "preset": self.preset.to_dict(),
            "score": self.score,
            "match_reason": self.match_reason,
        }


class PresetCatalog:
    """
    Central catalog for voice presets.

    Loads presets from JSON catalog files and provides:
    - Filtering by attributes (gender, age, accent, etc.)
    - Tag-based search
    - Semantic search (when sentence-transformers available)
    - Preset application to audio

    Example:
        >>> catalog = PresetCatalog()
        >>> catalog.load()
        >>>
        >>> # Get a specific preset
        >>> preset = catalog.get("qwen3:ryan")
        >>>
        >>> # Search by attributes
        >>> presets = catalog.filter(gender="female", energy="calm")
        >>>
        >>> # Semantic search
        >>> results = catalog.search("warm narrator for meditation")
    """

    def __init__(self, catalog_dir: Optional[Path] = None):
        """
        Initialize the catalog.

        Args:
            catalog_dir: Directory containing catalog JSON files.
                        Defaults to built-in catalog directory.
        """
        self.catalog_dir = catalog_dir or CATALOG_DIR
        self._presets: dict[str, VoicePreset] = {}
        self._metadata: dict[str, PresetCatalogMetadata] = {}
        self._search_index: Optional["SemanticSearch"] = None
        self._loaded = False

    def load(self, sources: Optional[list[str]] = None) -> "PresetCatalog":
        """
        Load presets from catalog files.

        Args:
            sources: List of source names to load (e.g., ["qwen3", "vibevoice"]).
                    If None, loads all available catalogs.

        Returns:
            Self for chaining.
        """
        if not self.catalog_dir.exists():
            logger.warning(f"Catalog directory not found: {self.catalog_dir}")
            return self

        # Find all JSON catalog files
        catalog_files = list(self.catalog_dir.glob("*.json"))

        for catalog_file in catalog_files:
            source_name = catalog_file.stem

            # Skip if sources specified and this isn't in list
            if sources and source_name not in sources:
                continue

            try:
                self._load_catalog_file(catalog_file)
            except Exception as e:
                logger.error(f"Failed to load catalog {catalog_file}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._presets)} presets from {len(catalog_files)} catalogs")
        return self

    def _load_catalog_file(self, path: Path) -> None:
        """Load a single catalog JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load metadata
        if "metadata" in data:
            source = data["metadata"].get("source", path.stem)
            self._metadata[source] = PresetCatalogMetadata.from_dict(data["metadata"])

        # Load presets
        for preset_data in data.get("presets", []):
            try:
                preset = VoicePreset.from_dict(preset_data)
                self._presets[preset.id] = preset
            except Exception as e:
                logger.warning(f"Failed to load preset {preset_data.get('id', 'unknown')}: {e}")

    def _ensure_loaded(self) -> None:
        """Ensure catalog is loaded."""
        if not self._loaded:
            self.load()

    # ===== Basic Access =====

    def get(self, preset_id: str) -> Optional[VoicePreset]:
        """
        Get a preset by ID.

        Args:
            preset_id: Preset ID (e.g., "qwen3:ryan")

        Returns:
            VoicePreset or None if not found.
        """
        self._ensure_loaded()
        return self._presets.get(preset_id)

    def get_or_raise(self, preset_id: str) -> VoicePreset:
        """Get a preset by ID, raising KeyError if not found."""
        preset = self.get(preset_id)
        if preset is None:
            raise KeyError(f"Preset not found: {preset_id}")
        return preset

    def all(self) -> list[VoicePreset]:
        """Get all presets."""
        self._ensure_loaded()
        return list(self._presets.values())

    def count(self) -> int:
        """Get total number of presets."""
        self._ensure_loaded()
        return len(self._presets)

    def sources(self) -> list[str]:
        """Get list of loaded sources."""
        self._ensure_loaded()
        return list(self._metadata.keys())

    def __iter__(self) -> Iterator[VoicePreset]:
        """Iterate over all presets."""
        self._ensure_loaded()
        return iter(self._presets.values())

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, preset_id: str) -> bool:
        self._ensure_loaded()
        return preset_id in self._presets

    # ===== Filtering =====

    def filter(
        self,
        source: Optional[str] = None,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        accent: Optional[str] = None,
        energy: Optional[str] = None,
        tone: Optional[str] = None,
        language: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_case: Optional[str] = None,
    ) -> list[VoicePreset]:
        """
        Filter presets by attributes.

        Args:
            source: Filter by source (e.g., "qwen3", "vibevoice")
            gender: Filter by gender ("male", "female", "neutral")
            age_range: Filter by age range ("child", "teen", "young_adult", etc.)
            accent: Filter by accent ("american", "british", "indian", etc.)
            energy: Filter by energy level ("calm", "neutral", "energetic")
            tone: Filter by tone ("warm", "neutral", "cool", "bright", "dark")
            language: Filter by supported language code
            tags: Filter by tags (preset must have ALL specified tags)
            use_case: Filter by use case

        Returns:
            List of matching presets.
        """
        self._ensure_loaded()

        results = []
        for preset in self._presets.values():
            # Source filter
            if source and preset.source.value != source:
                continue

            # Delegate to preset's filter method
            if not preset.matches_filters(
                gender=gender,
                age_range=age_range,
                accent=accent,
                energy=energy,
                tone=tone,
                language=language,
                tags=tags,
                use_case=use_case,
            ):
                continue

            results.append(preset)

        return results

    def by_source(self, source: str) -> list[VoicePreset]:
        """Get all presets from a specific source."""
        return self.filter(source=source)

    def by_gender(self, gender: str) -> list[VoicePreset]:
        """Get all presets matching a gender."""
        return self.filter(gender=gender)

    def by_use_case(self, use_case: str) -> list[VoicePreset]:
        """Get all presets suitable for a use case."""
        return self.filter(use_case=use_case)

    def by_tags(self, *tags: str) -> list[VoicePreset]:
        """Get all presets having ALL specified tags."""
        return self.filter(tags=list(tags))

    # ===== Tag-based Search =====

    def search_tags(self, query: str, limit: int = 10) -> list[SearchResult]:
        """
        Simple tag-based search.

        Searches preset names, descriptions, and tags for query terms.
        Falls back method when semantic search not available.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of SearchResult with relevance scores.
        """
        self._ensure_loaded()

        query_terms = query.lower().split()
        results = []

        for preset in self._presets.values():
            search_text = preset.get_search_text().lower()

            # Count matching terms
            matches = sum(1 for term in query_terms if term in search_text)

            if matches > 0:
                score = matches / len(query_terms)
                results.append(SearchResult(
                    preset=preset,
                    score=score,
                    match_reason=f"Matched {matches}/{len(query_terms)} terms",
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ===== Semantic Search =====

    def enable_semantic_search(self) -> bool:
        """
        Enable semantic search using sentence-transformers.

        Returns:
            True if enabled successfully, False if not available.
        """
        try:
            from .search import SemanticSearch
            self._ensure_loaded()
            self._search_index = SemanticSearch()
            self._search_index.index(self.all())
            logger.info("Semantic search enabled")
            return True
        except ImportError:
            logger.warning("sentence-transformers not installed, semantic search disabled")
            return False

    def search(
        self,
        query: str,
        limit: int = 5,
        gender: Optional[str] = None,
        age_range: Optional[str] = None,
        language: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for presets matching a natural language query.

        Uses semantic search if available, falls back to tag-based search.

        Args:
            query: Natural language query (e.g., "warm narrator for meditation")
            limit: Maximum results
            gender: Optional filter
            age_range: Optional filter
            language: Optional filter

        Returns:
            List of SearchResult with relevance scores.
        """
        self._ensure_loaded()

        # Try semantic search first
        if self._search_index is not None:
            results = self._search_index.search(query, top_k=limit * 2)
        else:
            results = self.search_tags(query, limit=limit * 2)

        # Apply filters
        filtered = []
        for result in results:
            if result.preset.matches_filters(
                gender=gender,
                age_range=age_range,
                language=language,
            ):
                filtered.append(result)

        return filtered[:limit]

    # ===== Utility Methods =====

    def list_tags(self) -> list[str]:
        """Get all unique tags across all presets."""
        self._ensure_loaded()
        tags = set()
        for preset in self._presets.values():
            tags.update(preset.tags)
        return sorted(tags)

    def list_use_cases(self) -> list[str]:
        """Get all unique use cases across all presets."""
        self._ensure_loaded()
        use_cases = set()
        for preset in self._presets.values():
            use_cases.update(preset.use_cases)
        return sorted(use_cases)

    def list_accents(self) -> list[str]:
        """Get all unique accents across all presets."""
        self._ensure_loaded()
        accents = set()
        for preset in self._presets.values():
            if preset.accent:
                accents.add(preset.accent)
        return sorted(accents)

    def demographics_summary(self) -> dict:
        """Get a summary of demographic coverage."""
        self._ensure_loaded()

        summary = {
            "total": len(self._presets),
            "by_source": {},
            "by_gender": {"male": 0, "female": 0, "neutral": 0},
            "by_age": {},
            "by_accent": {},
        }

        for preset in self._presets.values():
            # By source
            source = preset.source.value
            summary["by_source"][source] = summary["by_source"].get(source, 0) + 1

            # By gender
            if preset.gender:
                summary["by_gender"][preset.gender.value] = (
                    summary["by_gender"].get(preset.gender.value, 0) + 1
                )

            # By age
            if preset.age_range:
                age = preset.age_range.value
                summary["by_age"][age] = summary["by_age"].get(age, 0) + 1

            # By accent
            if preset.accent:
                summary["by_accent"][preset.accent] = (
                    summary["by_accent"].get(preset.accent, 0) + 1
                )

        return summary

    def to_compact_list(self) -> list[dict]:
        """
        Get a compact list of all presets for display.

        Returns minimal info suitable for listing/selection.
        """
        self._ensure_loaded()
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description[:100] + "..." if len(p.description) > 100 else p.description,
                "gender": p.gender.value if p.gender else None,
                "age": p.age_range.value if p.age_range else None,
                "accent": p.accent,
                "tags": p.tags[:5],  # First 5 tags
            }
            for p in self._presets.values()
        ]

    # ===== Preset Creation =====

    def create_custom(
        self,
        name: str,
        description: str,
        base_preset_id: Optional[str] = None,
        **kwargs,
    ) -> VoicePreset:
        """
        Create a custom preset.

        Args:
            name: Preset name
            description: Preset description
            base_preset_id: Optional base preset to inherit from
            **kwargs: Additional preset attributes

        Returns:
            New VoicePreset instance.
        """
        # Start with defaults or base preset
        if base_preset_id:
            base = self.get_or_raise(base_preset_id)
            preset_dict = base.to_dict()
        else:
            preset_dict = {
                "source": "custom",
                "tags": [],
                "use_cases": [],
                "language": "en",
                "languages": ["en"],
            }

        # Override with provided values
        preset_dict["id"] = f"custom:{name.lower().replace(' ', '_')}"
        preset_dict["name"] = name
        preset_dict["description"] = description
        preset_dict["source"] = "custom"
        preset_dict.update(kwargs)

        return VoicePreset.from_dict(preset_dict)

    def save_custom(self, preset: VoicePreset, path: Optional[Path] = None) -> Path:
        """
        Save a custom preset to JSON.

        Args:
            preset: Preset to save
            path: Output path. Defaults to catalog/custom.json

        Returns:
            Path where preset was saved.
        """
        if path is None:
            path = self.catalog_dir / "custom.json"

        # Load existing custom presets
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {
                "metadata": {
                    "source": "custom",
                    "version": "1.0",
                    "description": "User-created custom presets",
                    "author": "User",
                    "last_updated": "",
                    "preset_count": 0,
                },
                "presets": [],
            }

        # Add or update preset
        existing_ids = {p["id"] for p in data["presets"]}
        if preset.id in existing_ids:
            data["presets"] = [
                preset.to_dict() if p["id"] == preset.id else p
                for p in data["presets"]
            ]
        else:
            data["presets"].append(preset.to_dict())

        data["metadata"]["preset_count"] = len(data["presets"])

        # Save
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Reload to include new preset
        self._presets[preset.id] = preset

        return path


# Singleton instance for convenience
_catalog: Optional[PresetCatalog] = None


def get_catalog() -> PresetCatalog:
    """Get the global preset catalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = PresetCatalog()
        _catalog.load()
    return _catalog


def reload_catalog() -> PresetCatalog:
    """Reload the global catalog."""
    global _catalog
    _catalog = PresetCatalog()
    _catalog.load()
    return _catalog
