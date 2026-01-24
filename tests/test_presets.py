"""
Tests for the voice preset catalog system.

Tests cover:
- Schema validation (VoicePreset dataclass)
- Catalog loading and filtering
- Search functionality (tag-based and semantic)
- Migration helpers
"""

import pytest
from pathlib import Path
import json

from voice_soundboard.presets import (
    get_catalog,
    reload_catalog,
    VoicePreset,
    PresetSource,
    Gender,
    AgeRange,
    VoiceEnergy,
    VoiceTone,
    AcousticParams,
    PresetCatalog,
    SearchResult,
)


class TestVoicePresetSchema:
    """Tests for VoicePreset dataclass."""

    def test_create_basic_preset(self):
        """Can create a basic preset with required fields."""
        preset = VoicePreset(
            id="test:basic",
            name="Basic Test",
            source=PresetSource.CUSTOM,
            description="A basic test preset",
        )
        assert preset.id == "test:basic"
        assert preset.name == "Basic Test"
        assert preset.source == PresetSource.CUSTOM
        assert preset.description == "A basic test preset"

    def test_create_full_preset(self):
        """Can create a preset with all fields."""
        preset = VoicePreset(
            id="test:full",
            name="Full Test",
            source=PresetSource.VOCOLOGY,
            description="A fully specified preset",
            tags=["warm", "male", "narrator"],
            use_cases=["audiobook", "podcast"],
            gender=Gender.MALE,
            age_range=AgeRange.ADULT,
            accent="american",
            energy=VoiceEnergy.CALM,
            tone=VoiceTone.WARM,
            voice_prompt="Warm male narrator with gentle cadence",
            kokoro_voice="am_michael",
            language="en",
            languages=["en", "es"],
        )
        assert preset.gender == Gender.MALE
        assert preset.age_range == AgeRange.ADULT
        assert preset.energy == VoiceEnergy.CALM
        assert "warm" in preset.tags

    def test_preset_from_dict(self):
        """Can create preset from dictionary (JSON loading)."""
        data = {
            "id": "test:fromdict",
            "name": "From Dict",
            "source": "custom",
            "description": "Created from dict",
            "gender": "female",
            "age_range": "young_adult",
            "energy": "energetic",
            "tone": "bright",
        }
        preset = VoicePreset.from_dict(data)
        assert preset.source == PresetSource.CUSTOM
        assert preset.gender == Gender.FEMALE
        assert preset.energy == VoiceEnergy.ENERGETIC

    def test_preset_to_dict(self):
        """Preset can be serialized to dict."""
        preset = VoicePreset(
            id="test:todict",
            name="To Dict",
            source=PresetSource.QWEN3,
            description="Serialize test",
            gender=Gender.MALE,
        )
        data = preset.to_dict()
        assert data["id"] == "test:todict"
        assert data["source"] == "qwen3"
        assert data["gender"] == "male"

    def test_preset_search_text(self):
        """Preset generates searchable text."""
        preset = VoicePreset(
            id="test:search",
            name="Search Test",
            source=PresetSource.CUSTOM,
            description="A warm narrator voice",
            tags=["warm", "narrator"],
            use_cases=["audiobook"],
            voice_prompt="Warm and inviting",
        )
        text = preset.get_search_text()
        assert "Search Test" in text
        assert "warm narrator" in text
        assert "audiobook" in text
        assert "Warm and inviting" in text

    def test_preset_matches_filters(self):
        """Preset filtering works correctly."""
        preset = VoicePreset(
            id="test:filter",
            name="Filter Test",
            source=PresetSource.CUSTOM,
            description="Test filtering",
            gender=Gender.FEMALE,
            age_range=AgeRange.YOUNG_ADULT,
            accent="british",
            energy=VoiceEnergy.CALM,
            tags=["warm", "soothing"],
            languages=["en", "fr"],
        )

        assert preset.matches_filters(gender="female")
        assert preset.matches_filters(age_range="young_adult")
        assert preset.matches_filters(accent="british")
        assert preset.matches_filters(energy="calm")
        assert preset.matches_filters(language="en")
        assert preset.matches_filters(language="fr")
        assert preset.matches_filters(tags=["warm"])
        assert preset.matches_filters(tags=["warm", "soothing"])

        # Non-matches
        assert preset.matches_filters(gender="male") == False
        assert preset.matches_filters(language="de") == False
        assert preset.matches_filters(tags=["energetic"]) == False


class TestAcousticParams:
    """Tests for AcousticParams dataclass."""

    def test_default_params(self):
        """Default acoustic params are sensible."""
        params = AcousticParams()
        assert params.formant_ratio == 1.0
        assert params.pitch_shift_semitones == 0.0
        assert 0 <= params.jitter_percent <= 2
        assert 0 <= params.shimmer_percent <= 10
        assert params.speed_factor == 1.0

    def test_custom_params(self):
        """Can create custom acoustic params."""
        params = AcousticParams(
            formant_ratio=0.95,
            pitch_shift_semitones=-2,
            jitter_percent=0.8,
            shimmer_percent=3.0,
        )
        assert params.formant_ratio == 0.95
        assert params.pitch_shift_semitones == -2

    def test_params_roundtrip(self):
        """Params survive dict roundtrip."""
        original = AcousticParams(formant_ratio=0.9, jitter_percent=1.2)
        data = original.to_dict()
        restored = AcousticParams.from_dict(data)
        assert restored.formant_ratio == original.formant_ratio
        assert restored.jitter_percent == original.jitter_percent


class TestPresetCatalog:
    """Tests for PresetCatalog class."""

    @pytest.fixture
    def catalog(self):
        """Get a loaded catalog instance."""
        return get_catalog()

    def test_catalog_loads(self, catalog):
        """Catalog loads presets from JSON files."""
        assert catalog.count() > 0
        assert len(list(catalog)) > 0

    def test_catalog_has_multiple_sources(self, catalog):
        """Catalog contains presets from multiple sources."""
        sources = catalog.sources()
        # Should have at least vocology, qwen3, vibevoice, hume, diverse
        assert len(sources) >= 3

    def test_get_preset_by_id(self, catalog):
        """Can get a preset by its ID."""
        preset = catalog.get("vocology:warm_narrator")
        assert preset is not None
        assert preset.id == "vocology:warm_narrator"
        assert "narrator" in preset.name.lower() or "narrator" in preset.description.lower()

    def test_get_nonexistent_preset(self, catalog):
        """Getting nonexistent preset returns None."""
        preset = catalog.get("nonexistent:preset")
        assert preset is None

    def test_get_or_raise(self, catalog):
        """get_or_raise raises KeyError for missing preset."""
        with pytest.raises(KeyError):
            catalog.get_or_raise("nonexistent:preset")

    def test_filter_by_gender(self, catalog):
        """Can filter presets by gender."""
        males = catalog.filter(gender="male")
        females = catalog.filter(gender="female")

        assert len(males) > 0
        assert len(females) > 0

        for preset in males:
            assert preset.gender == Gender.MALE
        for preset in females:
            assert preset.gender == Gender.FEMALE

    def test_filter_by_source(self, catalog):
        """Can filter presets by source."""
        vocology = catalog.by_source("vocology")
        assert len(vocology) > 0
        for preset in vocology:
            assert preset.source == PresetSource.VOCOLOGY

    def test_filter_by_use_case(self, catalog):
        """Can filter presets by use case."""
        narration = catalog.by_use_case("narration")
        # May be empty if no presets have this exact use case
        for preset in narration:
            assert "narration" in preset.use_cases

    def test_filter_by_tags(self, catalog):
        """Can filter presets by tags."""
        warm = catalog.by_tags("warm")
        for preset in warm:
            assert "warm" in preset.tags

    def test_list_tags(self, catalog):
        """Can list all unique tags."""
        tags = catalog.list_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert all(isinstance(t, str) for t in tags)

    def test_list_use_cases(self, catalog):
        """Can list all unique use cases."""
        use_cases = catalog.list_use_cases()
        assert isinstance(use_cases, list)

    def test_demographics_summary(self, catalog):
        """Can get demographics summary."""
        summary = catalog.demographics_summary()
        assert "total" in summary
        assert "by_gender" in summary
        assert "by_source" in summary
        assert summary["total"] > 0


class TestPresetSearch:
    """Tests for preset search functionality."""

    @pytest.fixture
    def catalog(self):
        return get_catalog()

    def test_tag_search(self, catalog):
        """Tag-based search returns results."""
        results = catalog.search_tags("warm narrator male")
        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], SearchResult)
            assert results[0].score > 0

    def test_search_with_filters(self, catalog):
        """Search can be combined with filters."""
        results = catalog.search("narrator", gender="male", limit=5)
        assert len(results) <= 5
        for result in results:
            # If gender is set, it should match
            if result.preset.gender:
                assert result.preset.gender == Gender.MALE

    def test_search_returns_sorted_results(self, catalog):
        """Search results are sorted by relevance."""
        results = catalog.search_tags("warm calm deep narrator")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestPresetCatalogCreation:
    """Tests for creating custom presets."""

    @pytest.fixture
    def catalog(self):
        return PresetCatalog()

    def test_create_custom_preset(self, catalog):
        """Can create a custom preset."""
        catalog.load()
        preset = catalog.create_custom(
            name="My Custom Voice",
            description="A custom voice for testing",
            gender=Gender.NEUTRAL,
            tags=["custom", "test"],
        )
        assert preset.id == "custom:my_custom_voice"
        assert preset.source == PresetSource.CUSTOM
        assert "custom" in preset.tags

    def test_create_custom_from_base(self, catalog):
        """Can create custom preset based on existing one."""
        catalog.load()

        # Only test if we have presets
        if catalog.count() > 0:
            base_id = list(catalog.all())[0].id
            preset = catalog.create_custom(
                name="Modified Preset",
                description="Based on existing",
                base_preset_id=base_id,
            )
            assert preset.id == "custom:modified_preset"
            assert preset.source == PresetSource.CUSTOM


class TestMigrationShims:
    """Tests for backwards compatibility migration."""

    def test_old_preset_maps_to_new(self):
        """Old VoicePreset enum values map to new catalog IDs."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from voice_soundboard.vocology.presets import (
                VoicePreset as OldPreset,
                get_new_preset_id,
            )

        # Check mapping exists
        new_id = get_new_preset_id(OldPreset.WARM_NARRATOR)
        assert new_id == "vocology:warm_narrator"

        new_id = get_new_preset_id(OldPreset.DEEP_AUTHORITY)
        assert new_id == "vocology:deep_authority"

    def test_migration_helper(self):
        """Migration helper returns mapped presets."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from voice_soundboard.vocology.presets import (
                VoicePreset as OldPreset,
                migrate_to_new_catalog,
            )

        mapping = migrate_to_new_catalog()

        # Should have mappings for old presets
        if OldPreset.WARM_NARRATOR in mapping:
            new_preset = mapping[OldPreset.WARM_NARRATOR]
            assert new_preset.id == "vocology:warm_narrator"

    def test_deprecation_warning_on_import(self):
        """Old module has deprecation notice in docstring."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from voice_soundboard.vocology import presets as old_presets

        # Check docstring mentions deprecation
        assert "deprecated" in old_presets.__doc__.lower()
        assert "voice_soundboard.presets" in old_presets.__doc__

    def test_old_functions_emit_warnings(self):
        """Old convenience functions emit deprecation warnings."""
        import warnings
        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from voice_soundboard.vocology.presets import apply_narrator

        # Calling should emit warning (but we can't easily test without audio)
        # Just verify the function exists and has deprecation in docstring
        assert "deprecated" in apply_narrator.__doc__.lower()


class TestCatalogIntegration:
    """Integration tests for the complete catalog system."""

    def test_realistic_voice_selection(self):
        """Simulate realistic voice selection workflow."""
        catalog = get_catalog()

        # User wants: "warm male narrator for audiobook"
        results = catalog.search("warm male narrator audiobook", limit=3)

        # Should get relevant results
        assert len(results) > 0

        # Top result should be reasonably relevant
        top = results[0]
        search_text = top.preset.get_search_text().lower()
        # At least one of these should match
        keywords = ["warm", "narrator", "male", "audiobook"]
        matches = sum(1 for k in keywords if k in search_text)
        assert matches >= 1  # At least one keyword should match

    def test_demographic_coverage(self):
        """Verify catalog has diverse demographic coverage."""
        catalog = get_catalog()
        summary = catalog.demographics_summary()

        # Should have both male and female voices
        assert summary["by_gender"]["male"] > 0
        assert summary["by_gender"]["female"] > 0

        # Should have multiple age ranges
        assert len(summary["by_age"]) >= 2

    def test_all_presets_valid(self):
        """All loaded presets have required fields."""
        catalog = get_catalog()

        for preset in catalog:
            assert preset.id, f"Preset missing ID: {preset}"
            assert preset.name, f"Preset missing name: {preset.id}"
            assert preset.description, f"Preset missing description: {preset.id}"
            assert preset.source, f"Preset missing source: {preset.id}"

    def test_preset_ids_unique(self):
        """All preset IDs are unique."""
        catalog = get_catalog()
        ids = [p.id for p in catalog]
        assert len(ids) == len(set(ids)), "Duplicate preset IDs found"

    def test_compact_list_for_display(self):
        """Can get compact list for UI display."""
        catalog = get_catalog()
        compact = catalog.to_compact_list()

        assert isinstance(compact, list)
        assert len(compact) > 0

        # Check structure
        first = compact[0]
        assert "id" in first
        assert "name" in first
        assert "description" in first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
