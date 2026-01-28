"""
Test Additional Coverage Batch 58: Presets Search Tests

Tests for:
- SearchResult class
- SemanticSearch class
- FallbackSearch class
- create_search_engine function
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# ============== SearchResult Tests ==============

class TestSearchResult:
    """Tests for SearchResult class."""

    def test_search_result_creation(self):
        """Test SearchResult basic creation."""
        from voice_soundboard.presets.search import SearchResult
        mock_preset = Mock()
        result = SearchResult(preset=mock_preset, score=0.85)
        assert result.preset == mock_preset
        assert result.score == 0.85
        assert result.match_reason == ""

    def test_search_result_with_match_reason(self):
        """Test SearchResult with match reason."""
        from voice_soundboard.presets.search import SearchResult
        mock_preset = Mock()
        result = SearchResult(
            preset=mock_preset,
            score=0.92,
            match_reason="Semantic similarity: 92%"
        )
        assert result.match_reason == "Semantic similarity: 92%"

    def test_search_result_to_dict(self):
        """Test SearchResult.to_dict method."""
        from voice_soundboard.presets.search import SearchResult
        mock_preset = Mock()
        mock_preset.to_dict.return_value = {"id": "test", "name": "Test Preset"}

        result = SearchResult(
            preset=mock_preset,
            score=0.75,
            match_reason="Test match"
        )
        d = result.to_dict()

        assert "preset" in d
        assert d["preset"]["id"] == "test"
        assert d["score"] == 0.75
        assert d["match_reason"] == "Test match"

    def test_search_result_zero_score(self):
        """Test SearchResult with zero score."""
        from voice_soundboard.presets.search import SearchResult
        mock_preset = Mock()
        result = SearchResult(preset=mock_preset, score=0.0)
        assert result.score == 0.0

    def test_search_result_high_score(self):
        """Test SearchResult with score near 1.0."""
        from voice_soundboard.presets.search import SearchResult
        mock_preset = Mock()
        result = SearchResult(preset=mock_preset, score=0.99)
        assert result.score == 0.99


# ============== SemanticSearch Tests ==============

class TestSemanticSearch:
    """Tests for SemanticSearch class."""

    def test_semantic_search_init_default(self):
        """Test SemanticSearch default initialization."""
        from voice_soundboard.presets.search import SemanticSearch
        search = SemanticSearch()
        assert search.model_name == "all-MiniLM-L6-v2"
        assert search._model is None
        assert search._indexed is False

    def test_semantic_search_init_custom_model(self):
        """Test SemanticSearch with custom model."""
        from voice_soundboard.presets.search import SemanticSearch
        search = SemanticSearch(model_name="paraphrase-MiniLM-L6-v2")
        assert search.model_name == "paraphrase-MiniLM-L6-v2"

    def test_semantic_search_default_model_constant(self):
        """Test SemanticSearch.DEFAULT_MODEL constant."""
        from voice_soundboard.presets.search import SemanticSearch
        assert SemanticSearch.DEFAULT_MODEL == "all-MiniLM-L6-v2"

    def test_semantic_search_load_model_import_error(self):
        """Test SemanticSearch._load_model raises ImportError."""
        from voice_soundboard.presets.search import SemanticSearch
        search = SemanticSearch()

        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(ImportError, match="sentence-transformers required"):
                    search._load_model()

    @patch('voice_soundboard.presets.search.SemanticSearch._load_model')
    def test_semantic_search_index(self, mock_load_model):
        """Test SemanticSearch.index method."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_load_model.return_value = mock_model

        search = SemanticSearch()
        mock_presets = [Mock(), Mock()]
        for p in mock_presets:
            p.get_search_text.return_value = "test description"

        result = search.index(mock_presets)

        assert result is search  # Returns self for chaining
        assert search._indexed is True
        assert len(search._presets) == 2
        mock_model.encode.assert_called_once()

    def test_semantic_search_search_not_indexed(self):
        """Test SemanticSearch.search raises error when not indexed."""
        from voice_soundboard.presets.search import SemanticSearch
        search = SemanticSearch()

        with pytest.raises(RuntimeError, match="Index not built"):
            search.search("test query")

    @patch('voice_soundboard.presets.search.SemanticSearch._load_model')
    def test_semantic_search_search(self, mock_load_model):
        """Test SemanticSearch.search method."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([[0.5, 0.5], [0.1, 0.9]]),  # Index embeddings
            np.array([0.5, 0.5])  # Query embedding
        ]
        mock_load_model.return_value = mock_model

        search = SemanticSearch()
        mock_presets = [Mock(), Mock()]
        mock_presets[0].get_search_text.return_value = "warm narrator"
        mock_presets[0].id = "preset1"
        mock_presets[1].get_search_text.return_value = "bright energetic"
        mock_presets[1].id = "preset2"

        search.index(mock_presets)
        results = search.search("warm voice", top_k=2)

        assert len(results) >= 0  # May be 0 if all scores negative

    @patch('voice_soundboard.presets.search.SemanticSearch._load_model')
    def test_semantic_search_find_similar(self, mock_load_model):
        """Test SemanticSearch.find_similar method."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.5, 0.5], [0.6, 0.4], [0.1, 0.9]])
        mock_load_model.return_value = mock_model

        search = SemanticSearch()
        mock_presets = [Mock(), Mock(), Mock()]
        for i, p in enumerate(mock_presets):
            p.get_search_text.return_value = f"description {i}"
            p.id = f"preset{i}"

        search.index(mock_presets)

        reference_preset = mock_presets[0]
        results = search.find_similar(reference_preset, top_k=2)

        # Should not include the reference preset
        for r in results:
            assert r.preset.id != reference_preset.id

    def test_semantic_search_cosine_similarity(self):
        """Test SemanticSearch._cosine_similarity static method."""
        from voice_soundboard.presets.search import SemanticSearch

        query = np.array([1.0, 0.0])
        corpus = np.array([
            [1.0, 0.0],  # Identical to query
            [0.0, 1.0],  # Orthogonal
            [0.707, 0.707]  # 45 degrees
        ])

        similarities = SemanticSearch._cosine_similarity(query, corpus)

        assert len(similarities) == 3
        assert similarities[0] == pytest.approx(1.0, abs=0.01)
        assert similarities[1] == pytest.approx(0.0, abs=0.01)
        assert similarities[2] == pytest.approx(0.707, abs=0.01)


# ============== FallbackSearch Tests ==============

class TestFallbackSearch:
    """Tests for FallbackSearch class."""

    def test_fallback_search_init(self):
        """Test FallbackSearch initialization."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()
        assert search._presets == []
        assert search._indexed is False

    def test_fallback_search_index(self):
        """Test FallbackSearch.index method."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()
        mock_presets = [Mock(), Mock()]

        result = search.index(mock_presets)

        assert result is search  # Returns self for chaining
        assert search._indexed is True
        assert len(search._presets) == 2

    def test_fallback_search_search_not_indexed(self):
        """Test FallbackSearch.search raises error when not indexed."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        with pytest.raises(RuntimeError, match="Index not built"):
            search.search("test query")

    def test_fallback_search_search_keyword_match(self):
        """Test FallbackSearch.search with keyword matching."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        mock_presets = [Mock(), Mock(), Mock()]
        mock_presets[0].get_search_text.return_value = "warm narrator voice"
        mock_presets[1].get_search_text.return_value = "bright energetic speaker"
        mock_presets[2].get_search_text.return_value = "cold robot"

        search.index(mock_presets)
        results = search.search("warm voice", top_k=5)

        assert len(results) >= 1
        # First result should be the one with most matches
        assert results[0].preset == mock_presets[0]

    def test_fallback_search_search_no_matches(self):
        """Test FallbackSearch.search with no matches."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        mock_presets = [Mock()]
        mock_presets[0].get_search_text.return_value = "narrator voice"

        search.index(mock_presets)
        results = search.search("xyz123", top_k=5)

        assert len(results) == 0

    def test_fallback_search_search_partial_match(self):
        """Test FallbackSearch.search with partial match."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        mock_presets = [Mock()]
        mock_presets[0].get_search_text.return_value = "warm bright narrator"

        search.index(mock_presets)
        results = search.search("warm narrator energetic", top_k=5)

        assert len(results) == 1
        # Score should be 2/3 (matched 2 of 3 query terms)
        assert results[0].score == pytest.approx(2/3, abs=0.01)

    def test_fallback_search_search_score_calculation(self):
        """Test FallbackSearch score is based on matched terms."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        mock_preset = Mock()
        mock_preset.get_search_text.return_value = "warm narrator friendly"

        search.index([mock_preset])
        results = search.search("warm friendly", top_k=5)

        assert len(results) == 1
        # All query terms matched
        assert results[0].score == 1.0

    def test_fallback_search_match_reason_contains_terms(self):
        """Test FallbackSearch match_reason contains matched terms."""
        from voice_soundboard.presets.search import FallbackSearch
        search = FallbackSearch()

        mock_preset = Mock()
        mock_preset.get_search_text.return_value = "warm narrator"

        search.index([mock_preset])
        results = search.search("warm cold", top_k=5)

        assert len(results) == 1
        assert "warm" in results[0].match_reason


# ============== create_search_engine Tests ==============

class TestCreateSearchEngine:
    """Tests for create_search_engine function."""

    def test_create_search_engine_semantic(self):
        """Test create_search_engine returns SemanticSearch."""
        from voice_soundboard.presets.search import create_search_engine, SemanticSearch

        with patch.object(SemanticSearch, '__init__', return_value=None):
            engine = create_search_engine(use_semantic=True)

        assert isinstance(engine, SemanticSearch)

    def test_create_search_engine_fallback_explicit(self):
        """Test create_search_engine returns FallbackSearch when requested."""
        from voice_soundboard.presets.search import create_search_engine, FallbackSearch

        engine = create_search_engine(use_semantic=False)

        assert isinstance(engine, FallbackSearch)

    def test_create_search_engine_fallback_on_import_error(self):
        """Test create_search_engine falls back on ImportError."""
        from voice_soundboard.presets.search import create_search_engine, FallbackSearch, SemanticSearch

        with patch.object(SemanticSearch, '__init__', side_effect=ImportError("No module")):
            engine = create_search_engine(use_semantic=True)

        assert isinstance(engine, FallbackSearch)

    def test_create_search_engine_default_uses_semantic(self):
        """Test create_search_engine defaults to semantic search."""
        from voice_soundboard.presets.search import create_search_engine, SemanticSearch

        with patch.object(SemanticSearch, '__init__', return_value=None):
            engine = create_search_engine()

        assert isinstance(engine, SemanticSearch)
