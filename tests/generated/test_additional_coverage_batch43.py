"""
Additional coverage tests - Batch 43: Presets/Catalog Coverage.

Comprehensive tests for:
- voice_soundboard/presets/schema.py
- voice_soundboard/presets/catalog.py
- voice_soundboard/presets/search.py
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for preset enums."""

    def test_gender_enum(self):
        """Test Gender enum values."""
        from voice_soundboard.presets.schema import Gender

        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NEUTRAL.value == "neutral"

    def test_age_range_enum(self):
        """Test AgeRange enum values."""
        from voice_soundboard.presets.schema import AgeRange

        assert AgeRange.CHILD.value == "child"
        assert AgeRange.TEEN.value == "teen"
        assert AgeRange.YOUNG_ADULT.value == "young_adult"
        assert AgeRange.ADULT.value == "adult"
        assert AgeRange.ELDERLY.value == "elderly"

    def test_voice_energy_enum(self):
        """Test VoiceEnergy enum values."""
        from voice_soundboard.presets.schema import VoiceEnergy

        assert VoiceEnergy.CALM.value == "calm"
        assert VoiceEnergy.NEUTRAL.value == "neutral"
        assert VoiceEnergy.ENERGETIC.value == "energetic"

    def test_voice_tone_enum(self):
        """Test VoiceTone enum values."""
        from voice_soundboard.presets.schema import VoiceTone

        assert VoiceTone.WARM.value == "warm"
        assert VoiceTone.NEUTRAL.value == "neutral"
        assert VoiceTone.COOL.value == "cool"

    def test_preset_source_enum(self):
        """Test PresetSource enum values."""
        from voice_soundboard.presets.schema import PresetSource

        assert PresetSource.QWEN3.value == "qwen3"
        assert PresetSource.KOKORO.value == "kokoro"
        assert PresetSource.CUSTOM.value == "custom"


# =============================================================================
# AcousticParams Tests
# =============================================================================

class TestAcousticParams:
    """Tests for AcousticParams dataclass."""

    def test_acoustic_params_defaults(self):
        """Test AcousticParams default values."""
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams()
        assert params.formant_ratio == 1.0
        assert params.pitch_shift_semitones == 0.0
        assert params.speed_factor == 1.0

    def test_acoustic_params_custom(self):
        """Test AcousticParams with custom values."""
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(
            formant_ratio=1.2,
            pitch_shift_semitones=2.0,
            jitter_percent=1.0,
        )
        assert params.formant_ratio == 1.2
        assert params.pitch_shift_semitones == 2.0

    def test_acoustic_params_to_dict(self):
        """Test AcousticParams to_dict method."""
        from voice_soundboard.presets.schema import AcousticParams

        params = AcousticParams(formant_ratio=1.1)
        data = params.to_dict()

        assert data["formant_ratio"] == 1.1
        assert "pitch_shift_semitones" in data
        assert "emotional_state" in data

    def test_acoustic_params_from_dict(self):
        """Test AcousticParams from_dict method."""
        from voice_soundboard.presets.schema import AcousticParams

        data = {
            "formant_ratio": 0.9,
            "speed_factor": 1.2,
            "emotional_state": "excited",
        }
        params = AcousticParams.from_dict(data)

        assert params.formant_ratio == 0.9
        assert params.speed_factor == 1.2
        assert params.emotional_state == "excited"


# =============================================================================
# VoicePreset Tests
# =============================================================================

class TestVoicePreset:
    """Tests for VoicePreset dataclass."""

    def test_preset_creation(self):
        """Test creating a VoicePreset."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource, Gender

        preset = VoicePreset(
            id="test:voice",
            name="Test Voice",
            source=PresetSource.CUSTOM,
            description="A test voice preset",
        )
        assert preset.id == "test:voice"
        assert preset.name == "Test Voice"

    def test_preset_with_all_fields(self):
        """Test VoicePreset with all fields."""
        from voice_soundboard.presets.schema import (
            VoicePreset, PresetSource, Gender, AgeRange, VoiceEnergy, VoiceTone
        )

        preset = VoicePreset(
            id="full:preset",
            name="Full Preset",
            source=PresetSource.QWEN3,
            description="A fully configured preset",
            tags=["warm", "female", "narrator"],
            use_cases=["podcast", "audiobook"],
            gender=Gender.FEMALE,
            age_range=AgeRange.ADULT,
            accent="american",
            energy=VoiceEnergy.CALM,
            tone=VoiceTone.WARM,
            language="en",
        )

        assert preset.gender == Gender.FEMALE
        assert preset.age_range == AgeRange.ADULT
        assert "narrator" in preset.tags

    def test_preset_to_dict(self):
        """Test VoicePreset to_dict method."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        preset = VoicePreset(
            id="test:dict",
            name="Dict Test",
            source=PresetSource.CUSTOM,
            description="Test description",
            tags=["tag1", "tag2"],
        )
        data = preset.to_dict()

        assert data["id"] == "test:dict"
        assert data["name"] == "Dict Test"
        assert data["source"] == "custom"
        assert "tag1" in data["tags"]

    def test_preset_from_dict(self):
        """Test VoicePreset from_dict method."""
        from voice_soundboard.presets.schema import VoicePreset, Gender

        data = {
            "id": "loaded:preset",
            "name": "Loaded Preset",
            "source": "kokoro",
            "description": "Loaded from dict",
            "gender": "female",
            "tags": ["test"],
        }
        preset = VoicePreset.from_dict(data)

        assert preset.id == "loaded:preset"
        assert preset.gender == Gender.FEMALE

    def test_preset_get_search_text(self):
        """Test VoicePreset get_search_text method."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource, Gender

        preset = VoicePreset(
            id="search:test",
            name="Searchable Voice",
            source=PresetSource.CUSTOM,
            description="A warm, friendly voice for narration",
            tags=["warm", "friendly", "narrator"],
            gender=Gender.FEMALE,
            voice_prompt="Speak in a warm, welcoming tone",
        )
        search_text = preset.get_search_text()

        assert "Searchable Voice" in search_text
        assert "warm" in search_text
        assert "friendly" in search_text
        assert "narration" in search_text

    def test_preset_matches_filters_gender(self):
        """Test VoicePreset matches_filters for gender."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource, Gender

        preset = VoicePreset(
            id="filter:test",
            name="Filter Test",
            source=PresetSource.CUSTOM,
            description="Test",
            gender=Gender.FEMALE,
        )

        assert preset.matches_filters(gender="female") is True
        assert preset.matches_filters(gender="male") is False

    def test_preset_matches_filters_tags(self):
        """Test VoicePreset matches_filters for tags."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        preset = VoicePreset(
            id="tags:test",
            name="Tags Test",
            source=PresetSource.CUSTOM,
            description="Test",
            tags=["warm", "narrator", "calm"],
        )

        assert preset.matches_filters(tags=["warm"]) is True
        assert preset.matches_filters(tags=["warm", "narrator"]) is True
        assert preset.matches_filters(tags=["missing"]) is False

    def test_preset_matches_filters_use_case(self):
        """Test VoicePreset matches_filters for use_case."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        preset = VoicePreset(
            id="usecase:test",
            name="UseCase Test",
            source=PresetSource.CUSTOM,
            description="Test",
            use_cases=["podcast", "audiobook"],
        )

        assert preset.matches_filters(use_case="podcast") is True
        assert preset.matches_filters(use_case="gaming") is False

    def test_preset_summary(self):
        """Test VoicePreset summary method."""
        from voice_soundboard.presets.schema import VoicePreset, PresetSource, Gender

        preset = VoicePreset(
            id="summary:test",
            name="Summary Voice",
            source=PresetSource.CUSTOM,
            description="A long description that should be truncated",
            gender=Gender.MALE,
            accent="british",
        )
        summary = preset.summary()

        assert "Summary Voice" in summary
        assert "male" in summary
        assert "british" in summary

    def test_preset_post_init_string_enums(self):
        """Test that string values are converted to enums in __post_init__."""
        from voice_soundboard.presets.schema import VoicePreset, Gender, AgeRange

        preset = VoicePreset(
            id="string:test",
            name="String Test",
            source="custom",  # String instead of enum
            description="Test",
            gender="female",  # String instead of enum
            age_range="adult",  # String instead of enum
        )

        assert isinstance(preset.gender, Gender)
        assert isinstance(preset.age_range, AgeRange)


# =============================================================================
# PresetCatalogMetadata Tests
# =============================================================================

class TestPresetCatalogMetadata:
    """Tests for PresetCatalogMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating PresetCatalogMetadata."""
        from voice_soundboard.presets.schema import PresetCatalogMetadata

        metadata = PresetCatalogMetadata(
            source="test",
            version="1.0",
            description="Test catalog",
            author="Tester",
            last_updated="2024-01-01",
            preset_count=10,
        )
        assert metadata.source == "test"
        assert metadata.preset_count == 10

    def test_metadata_to_dict(self):
        """Test PresetCatalogMetadata to_dict method."""
        from voice_soundboard.presets.schema import PresetCatalogMetadata

        metadata = PresetCatalogMetadata(
            source="test",
            version="1.0",
            description="Test",
            author="Tester",
            last_updated="2024-01-01",
        )
        data = metadata.to_dict()

        assert data["source"] == "test"
        assert data["version"] == "1.0"

    def test_metadata_from_dict(self):
        """Test PresetCatalogMetadata from_dict method."""
        from voice_soundboard.presets.schema import PresetCatalogMetadata

        data = {
            "source": "loaded",
            "version": "2.0",
            "description": "Loaded metadata",
            "author": "Someone",
            "last_updated": "2024-06-01",
            "preset_count": 50,
        }
        metadata = PresetCatalogMetadata.from_dict(data)

        assert metadata.source == "loaded"
        assert metadata.preset_count == 50


# =============================================================================
# SearchResult Tests
# =============================================================================

class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        from voice_soundboard.presets.catalog import SearchResult
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        preset = VoicePreset(
            id="result:test",
            name="Result Test",
            source=PresetSource.CUSTOM,
            description="Test",
        )
        result = SearchResult(preset=preset, score=0.85, match_reason="Test match")

        assert result.preset.id == "result:test"
        assert result.score == 0.85
        assert result.match_reason == "Test match"

    def test_search_result_to_dict(self):
        """Test SearchResult to_dict method."""
        from voice_soundboard.presets.catalog import SearchResult
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        preset = VoicePreset(
            id="dict:test",
            name="Dict Test",
            source=PresetSource.CUSTOM,
            description="Test",
        )
        result = SearchResult(preset=preset, score=0.9)
        data = result.to_dict()

        assert data["score"] == 0.9
        assert "preset" in data


# =============================================================================
# PresetCatalog Tests
# =============================================================================

class TestPresetCatalog:
    """Tests for PresetCatalog class."""

    def test_catalog_creation(self, tmp_path):
        """Test creating a PresetCatalog."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog = PresetCatalog(catalog_dir=tmp_path)
        assert catalog is not None

    def test_catalog_load_empty_dir(self, tmp_path):
        """Test loading from empty directory."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()
        assert catalog.count() == 0

    def test_catalog_load_json_file(self, tmp_path):
        """Test loading catalog from JSON file."""
        from voice_soundboard.presets.catalog import PresetCatalog

        # Create test catalog file
        catalog_data = {
            "metadata": {
                "source": "test",
                "version": "1.0",
                "description": "Test catalog",
                "author": "Tester",
                "last_updated": "2024-01-01",
                "preset_count": 1,
            },
            "presets": [
                {
                    "id": "test:voice1",
                    "name": "Voice One",
                    "source": "custom",
                    "description": "First test voice",
                    "tags": ["test"],
                }
            ],
        }

        catalog_file = tmp_path / "test.json"
        with open(catalog_file, "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        assert catalog.count() == 1
        assert "test:voice1" in catalog

    def test_catalog_get(self, tmp_path):
        """Test getting a preset by ID."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:get", "name": "Get Test", "source": "custom", "description": "Test"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        preset = catalog.get("test:get")
        assert preset is not None
        assert preset.name == "Get Test"

        assert catalog.get("nonexistent") is None

    def test_catalog_get_or_raise(self, tmp_path):
        """Test get_or_raise method."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:raise", "name": "Raise Test", "source": "custom", "description": "Test"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        preset = catalog.get_or_raise("test:raise")
        assert preset.name == "Raise Test"

        with pytest.raises(KeyError):
            catalog.get_or_raise("nonexistent")

    def test_catalog_filter_gender(self, tmp_path):
        """Test filtering by gender."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:male", "name": "Male Voice", "source": "custom", "description": "Test", "gender": "male"},
                {"id": "test:female", "name": "Female Voice", "source": "custom", "description": "Test", "gender": "female"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        female_presets = catalog.filter(gender="female")
        assert len(female_presets) == 1
        assert female_presets[0].name == "Female Voice"

    def test_catalog_filter_tags(self, tmp_path):
        """Test filtering by tags."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:narrator", "name": "Narrator", "source": "custom", "description": "Test", "tags": ["narrator", "calm"]},
                {"id": "test:energetic", "name": "Energetic", "source": "custom", "description": "Test", "tags": ["energetic"]},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        narrator_presets = catalog.by_tags("narrator")
        assert len(narrator_presets) == 1

    def test_catalog_search_tags(self, tmp_path):
        """Test tag-based search."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:warm", "name": "Warm Voice", "source": "custom", "description": "A warm friendly voice", "tags": ["warm", "friendly"]},
                {"id": "test:cool", "name": "Cool Voice", "source": "custom", "description": "A cool distant voice", "tags": ["cool"]},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        results = catalog.search_tags("warm friendly")
        assert len(results) > 0
        assert results[0].preset.id == "test:warm"

    def test_catalog_list_tags(self, tmp_path):
        """Test listing all tags."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:1", "name": "One", "source": "custom", "description": "Test", "tags": ["tag1", "tag2"]},
                {"id": "test:2", "name": "Two", "source": "custom", "description": "Test", "tags": ["tag2", "tag3"]},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        tags = catalog.list_tags()
        assert "tag1" in tags
        assert "tag2" in tags
        assert "tag3" in tags

    def test_catalog_demographics_summary(self, tmp_path):
        """Test demographics summary."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:1", "name": "One", "source": "custom", "description": "Test", "gender": "male"},
                {"id": "test:2", "name": "Two", "source": "custom", "description": "Test", "gender": "female"},
                {"id": "test:3", "name": "Three", "source": "custom", "description": "Test", "gender": "female"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        summary = catalog.demographics_summary()
        assert summary["total"] == 3
        assert summary["by_gender"]["male"] == 1
        assert summary["by_gender"]["female"] == 2

    def test_catalog_iteration(self, tmp_path):
        """Test iterating over catalog."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:1", "name": "One", "source": "custom", "description": "Test"},
                {"id": "test:2", "name": "Two", "source": "custom", "description": "Test"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        presets = list(catalog)
        assert len(presets) == 2

    def test_catalog_contains(self, tmp_path):
        """Test __contains__ method."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog_data = {
            "metadata": {"source": "test", "version": "1.0", "description": "", "author": "", "last_updated": ""},
            "presets": [
                {"id": "test:contains", "name": "Contains", "source": "custom", "description": "Test"},
            ],
        }
        with open(tmp_path / "test.json", "w") as f:
            json.dump(catalog_data, f)

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        assert "test:contains" in catalog
        assert "nonexistent" not in catalog


# =============================================================================
# FallbackSearch Tests
# =============================================================================

class TestFallbackSearch:
    """Tests for FallbackSearch class."""

    def test_fallback_search_creation(self):
        """Test creating FallbackSearch."""
        from voice_soundboard.presets.search import FallbackSearch

        search = FallbackSearch()
        assert search is not None

    def test_fallback_search_index_and_search(self):
        """Test indexing and searching with FallbackSearch."""
        from voice_soundboard.presets.search import FallbackSearch
        from voice_soundboard.presets.schema import VoicePreset, PresetSource

        presets = [
            VoicePreset(
                id="test:warm",
                name="Warm Voice",
                source=PresetSource.CUSTOM,
                description="A warm friendly narrator voice",
            ),
            VoicePreset(
                id="test:cold",
                name="Cold Voice",
                source=PresetSource.CUSTOM,
                description="A cold distant robotic voice",
            ),
        ]

        search = FallbackSearch()
        search.index(presets)

        results = search.search("warm friendly")
        assert len(results) > 0
        assert results[0].preset.id == "test:warm"

    def test_fallback_search_no_index(self):
        """Test searching without indexing raises error."""
        from voice_soundboard.presets.search import FallbackSearch

        search = FallbackSearch()

        with pytest.raises(RuntimeError):
            search.search("test query")


# =============================================================================
# Catalog Helper Functions Tests
# =============================================================================

class TestCatalogHelperFunctions:
    """Tests for catalog helper functions."""

    def test_get_catalog(self):
        """Test get_catalog function."""
        from voice_soundboard.presets.catalog import get_catalog

        catalog = get_catalog()
        assert catalog is not None

    def test_reload_catalog(self):
        """Test reload_catalog function."""
        from voice_soundboard.presets.catalog import reload_catalog

        catalog = reload_catalog()
        assert catalog is not None


# =============================================================================
# Create Custom Preset Tests
# =============================================================================

class TestCreateCustomPreset:
    """Tests for creating custom presets."""

    def test_create_custom_preset(self, tmp_path):
        """Test creating a custom preset."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        custom = catalog.create_custom(
            name="My Custom Voice",
            description="A custom voice I created",
            tags=["custom", "test"],
        )

        assert custom.id == "custom:my_custom_voice"
        assert custom.name == "My Custom Voice"
        assert "custom" in custom.tags

    def test_save_custom_preset(self, tmp_path):
        """Test saving a custom preset."""
        from voice_soundboard.presets.catalog import PresetCatalog

        catalog = PresetCatalog(catalog_dir=tmp_path)
        catalog.load()

        custom = catalog.create_custom(
            name="Saved Custom",
            description="A saved custom voice",
        )

        save_path = catalog.save_custom(custom)
        assert save_path.exists()

        # Reload and verify
        catalog2 = PresetCatalog(catalog_dir=tmp_path)
        catalog2.load()
        assert "custom:saved_custom" in catalog2
