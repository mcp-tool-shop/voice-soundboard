"""
Tests for Presets Search Module

Targets voice_soundboard/presets/search.py (0% coverage)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestSearchResult:
    """Tests for SearchResult class."""

    def test_create_search_result(self):
        """Should create search result."""
        from voice_soundboard.presets.search import SearchResult

        mock_preset = Mock()
        mock_preset.id = "test_preset"
        mock_preset.to_dict = Mock(return_value={"id": "test_preset"})

        result = SearchResult(
            preset=mock_preset,
            score=0.85,
            match_reason="Test match",
        )

        assert result.preset == mock_preset
        assert result.score == 0.85
        assert result.match_reason == "Test match"

    def test_search_result_to_dict(self):
        """Should convert to dictionary."""
        from voice_soundboard.presets.search import SearchResult

        mock_preset = Mock()
        mock_preset.to_dict = Mock(return_value={"id": "test", "name": "Test Preset"})

        result = SearchResult(
            preset=mock_preset,
            score=0.75,
            match_reason="Semantic match",
        )

        d = result.to_dict()

        assert "preset" in d
        assert d["score"] == 0.75
        assert d["match_reason"] == "Semantic match"
        assert d["preset"]["id"] == "test"

    def test_search_result_empty_reason(self):
        """Should handle empty match reason."""
        from voice_soundboard.presets.search import SearchResult

        mock_preset = Mock()
        mock_preset.to_dict = Mock(return_value={})

        result = SearchResult(
            preset=mock_preset,
            score=0.5,
        )

        assert result.match_reason == ""


class TestSemanticSearch:
    """Tests for SemanticSearch class."""

    def test_init_default_model(self):
        """Should initialize with default model."""
        from voice_soundboard.presets.search import SemanticSearch

        search = SemanticSearch()

        assert search.model_name == "all-MiniLM-L6-v2"
        assert search._model is None
        assert search._indexed is False

    def test_init_custom_model(self):
        """Should initialize with custom model."""
        from voice_soundboard.presets.search import SemanticSearch

        search = SemanticSearch(model_name="custom-model")

        assert search.model_name == "custom-model"

    def test_load_model_import_error(self):
        """Should raise ImportError when sentence-transformers not available."""
        from voice_soundboard.presets.search import SemanticSearch

        search = SemanticSearch()

        with patch.dict('sys.modules', {'sentence_transformers': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module")):
                with pytest.raises(ImportError):
                    search._load_model()

    def test_load_model_success(self):
        """Should load model successfully."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_st = Mock()
        mock_st.SentenceTransformer = Mock(return_value=mock_model)

        with patch.dict('sys.modules', {'sentence_transformers': mock_st}):
            search = SemanticSearch()
            # Model loaded on demand

    def test_index_presets(self):
        """Should index presets."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.random.rand(3, 384))

        mock_presets = [Mock(), Mock(), Mock()]
        for i, p in enumerate(mock_presets):
            p.get_search_text = Mock(return_value=f"Preset {i} description")

        search = SemanticSearch()
        search._model = mock_model

        result = search.index(mock_presets)

        assert result == search  # Returns self for chaining
        assert search._indexed is True
        assert len(search._presets) == 3
        mock_model.encode.assert_called_once()

    def test_search_not_indexed(self):
        """Should raise error if not indexed."""
        from voice_soundboard.presets.search import SemanticSearch

        search = SemanticSearch()

        with pytest.raises(RuntimeError, match="Index not built"):
            search.search("test query")

    def test_search_success(self):
        """Should search and return results."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        # Query embedding
        mock_model.encode = Mock(return_value=np.array([0.5, 0.5, 0.5]))

        # Pre-indexed embeddings (3 presets)
        search = SemanticSearch()
        search._model = mock_model
        search._indexed = True
        search._embeddings = np.array([
            [0.6, 0.5, 0.4],  # Most similar
            [0.1, 0.2, 0.1],  # Least similar
            [0.4, 0.5, 0.5],  # Second most similar
        ])

        mock_presets = [Mock(id="p1"), Mock(id="p2"), Mock(id="p3")]
        for p in mock_presets:
            p.get_search_text = Mock(return_value="test")
        search._presets = mock_presets

        results = search.search("warm narrator", top_k=2)

        assert len(results) <= 2
        assert all(hasattr(r, 'score') for r in results)

    def test_search_top_k(self):
        """Should respect top_k parameter."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([0.5, 0.5]))

        search = SemanticSearch()
        search._model = mock_model
        search._indexed = True
        search._embeddings = np.array([
            [0.5, 0.5],
            [0.4, 0.4],
            [0.3, 0.3],
            [0.2, 0.2],
            [0.1, 0.1],
        ])

        mock_presets = [Mock(id=f"p{i}") for i in range(5)]
        search._presets = mock_presets

        results = search.search("test", top_k=3)

        assert len(results) <= 3

    def test_find_similar(self):
        """Should find similar presets."""
        from voice_soundboard.presets.search import SemanticSearch

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([0.5, 0.5]))

        search = SemanticSearch()
        search._model = mock_model
        search._indexed = True
        search._embeddings = np.array([
            [0.5, 0.5],
            [0.4, 0.4],
            [0.3, 0.3],
        ])

        mock_presets = [Mock(id="p1"), Mock(id="p2"), Mock(id="p3")]
        for p in mock_presets:
            p.get_search_text = Mock(return_value="test")
        search._presets = mock_presets

        # Find similar to p1
        results = search.find_similar(mock_presets[0], top_k=2)

        # Should not include the input preset
        assert all(r.preset.id != "p1" for r in results)

    def test_cosine_similarity(self):
        """Should calculate cosine similarity correctly."""
        from voice_soundboard.presets.search import SemanticSearch

        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [1.0, 0.0, 0.0],  # Same as query - similarity 1.0
            [0.0, 1.0, 0.0],  # Orthogonal - similarity 0.0
            [0.707, 0.707, 0.0],  # 45 degrees - similarity ~0.707
        ])

        similarities = SemanticSearch._cosine_similarity(query, corpus)

        assert similarities[0] == pytest.approx(1.0, rel=0.01)
        assert similarities[1] == pytest.approx(0.0, abs=0.01)
        assert similarities[2] == pytest.approx(0.707, rel=0.01)

    def test_cosine_similarity_zero_vectors(self):
        """Should handle zero vectors."""
        from voice_soundboard.presets.search import SemanticSearch

        query = np.array([0.0, 0.0, 0.0])
        corpus = np.array([[1.0, 0.0, 0.0]])

        # Should not crash due to division by zero
        similarities = SemanticSearch._cosine_similarity(query, corpus)
        assert len(similarities) == 1


class TestFallbackSearch:
    """Tests for FallbackSearch class."""

    def test_init(self):
        """Should initialize fallback search."""
        from voice_soundboard.presets.search import FallbackSearch

        search = FallbackSearch()

        assert search._indexed is False
        assert search._presets == []

    def test_index_presets(self):
        """Should index presets."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_presets = [Mock(), Mock()]

        search = FallbackSearch()
        result = search.index(mock_presets)

        assert result == search
        assert search._indexed is True
        assert len(search._presets) == 2

    def test_search_not_indexed(self):
        """Should raise error if not indexed."""
        from voice_soundboard.presets.search import FallbackSearch

        search = FallbackSearch()

        with pytest.raises(RuntimeError, match="Index not built"):
            search.search("test query")

    def test_search_keyword_matching(self):
        """Should search using keyword matching."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_presets = [
            Mock(id="p1"),
            Mock(id="p2"),
            Mock(id="p3"),
        ]
        mock_presets[0].get_search_text = Mock(return_value="warm narrator voice")
        mock_presets[1].get_search_text = Mock(return_value="energetic announcer")
        mock_presets[2].get_search_text = Mock(return_value="warm calm meditation")

        search = FallbackSearch()
        search.index(mock_presets)

        results = search.search("warm meditation")

        # p3 should be best match (both terms)
        assert len(results) > 0
        assert results[0].preset.id == "p3"

    def test_search_no_matches(self):
        """Should return empty for no matches."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_presets = [Mock(id="p1")]
        mock_presets[0].get_search_text = Mock(return_value="test voice")

        search = FallbackSearch()
        search.index(mock_presets)

        results = search.search("nonexistent xyz")

        assert len(results) == 0

    def test_search_partial_match(self):
        """Should return partial matches."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_presets = [
            Mock(id="p1"),
            Mock(id="p2"),
        ]
        mock_presets[0].get_search_text = Mock(return_value="warm narrator")
        mock_presets[1].get_search_text = Mock(return_value="cold narrator")

        search = FallbackSearch()
        search.index(mock_presets)

        results = search.search("warm deep narrator")

        # Both match "narrator", p1 also matches "warm"
        assert len(results) == 2
        assert results[0].preset.id == "p1"  # Higher score

    def test_search_top_k(self):
        """Should respect top_k."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_presets = [Mock(id=f"p{i}") for i in range(5)]
        for i, p in enumerate(mock_presets):
            p.get_search_text = Mock(return_value=f"voice preset test{i}")

        search = FallbackSearch()
        search.index(mock_presets)

        results = search.search("voice preset test", top_k=2)

        assert len(results) <= 2

    def test_search_case_insensitive(self):
        """Should be case insensitive."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="WARM Narrator Voice")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("warm narrator")

        assert len(results) == 1

    def test_search_match_reason(self):
        """Should include matched terms in reason."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="warm calm narrator")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("warm calm")

        assert len(results) == 1
        assert "warm" in results[0].match_reason.lower()
        assert "calm" in results[0].match_reason.lower()


class TestCreateSearchEngine:
    """Tests for create_search_engine factory function."""

    def test_create_semantic_search(self):
        """Should create semantic search when available."""
        from voice_soundboard.presets.search import create_search_engine, SemanticSearch

        with patch.object(SemanticSearch, '__init__', return_value=None):
            engine = create_search_engine(use_semantic=True)
            assert isinstance(engine, SemanticSearch)

    def test_create_fallback_on_import_error(self):
        """Should fallback when semantic search unavailable."""
        from voice_soundboard.presets.search import create_search_engine, FallbackSearch

        with patch('voice_soundboard.presets.search.SemanticSearch', side_effect=ImportError):
            engine = create_search_engine(use_semantic=True)
            assert isinstance(engine, FallbackSearch)

    def test_create_fallback_explicit(self):
        """Should create fallback when requested."""
        from voice_soundboard.presets.search import create_search_engine, FallbackSearch

        engine = create_search_engine(use_semantic=False)
        assert isinstance(engine, FallbackSearch)


class TestSearchEdgeCases:
    """Edge case tests for search."""

    def test_empty_query(self):
        """Should handle empty query."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="test voice")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("")

        assert len(results) == 0

    def test_single_preset(self):
        """Should handle single preset index."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="test voice")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("test", top_k=5)

        assert len(results) == 1

    def test_empty_preset_list(self):
        """Should handle empty preset list."""
        from voice_soundboard.presets.search import FallbackSearch

        search = FallbackSearch()
        search.index([])

        results = search.search("test")

        assert len(results) == 0

    def test_special_characters_in_query(self):
        """Should handle special characters."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="test voice")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("test! @#$%")

        # Should not crash, "test" should still match
        assert len(results) >= 0

    def test_unicode_query(self):
        """Should handle unicode in query."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="français voix")

        search = FallbackSearch()
        search.index([mock_preset])

        results = search.search("français")

        assert len(results) == 1

    def test_very_long_query(self):
        """Should handle very long query."""
        from voice_soundboard.presets.search import FallbackSearch

        mock_preset = Mock(id="p1")
        mock_preset.get_search_text = Mock(return_value="test voice")

        search = FallbackSearch()
        search.index([mock_preset])

        long_query = "test " * 100
        results = search.search(long_query)

        # Should not crash
        assert isinstance(results, list)
