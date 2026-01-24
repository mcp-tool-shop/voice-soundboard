"""
Semantic Search for Voice Presets

Uses sentence-transformers for embedding-based semantic search.
Falls back gracefully if not installed.
"""

import logging
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .schema import VoicePreset

logger = logging.getLogger(__name__)


class SearchResult:
    """A search result with relevance score."""

    def __init__(self, preset: "VoicePreset", score: float, match_reason: str = ""):
        self.preset = preset
        self.score = score
        self.match_reason = match_reason

    def to_dict(self) -> dict:
        return {
            "preset": self.preset.to_dict(),
            "score": self.score,
            "match_reason": self.match_reason,
        }


class SemanticSearch:
    """
    Semantic search engine for voice presets.

    Uses sentence-transformers to create embeddings of preset descriptions
    and finds similar presets based on cosine similarity.

    Example:
        >>> search = SemanticSearch()
        >>> search.index(presets)
        >>> results = search.search("warm narrator for meditation", top_k=5)
    """

    # Small, fast model that works well for short descriptions
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize semantic search.

        Args:
            model_name: Sentence transformer model name.
                       Defaults to all-MiniLM-L6-v2 (80MB, fast).
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._presets: list["VoicePreset"] = []
        self._embeddings: Optional[np.ndarray] = None
        self._indexed = False

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic search. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def index(self, presets: list["VoicePreset"]) -> "SemanticSearch":
        """
        Build search index from presets.

        Creates embeddings for all preset descriptions for fast similarity search.

        Args:
            presets: List of presets to index.

        Returns:
            Self for chaining.
        """
        model = self._load_model()
        self._presets = list(presets)

        # Create searchable text for each preset
        texts = [p.get_search_text() for p in self._presets]

        # Generate embeddings
        self._embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self._indexed = True
        logger.info(f"Indexed {len(self._presets)} presets for semantic search")
        return self

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search for presets matching a natural language query.

        Args:
            query: Natural language search query.
            top_k: Number of top results to return.

        Returns:
            List of SearchResult sorted by relevance (highest first).
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        model = self._load_model()

        # Encode query
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self._embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0:  # Only include positive matches
                results.append(SearchResult(
                    preset=self._presets[idx],
                    score=score,
                    match_reason=f"Semantic similarity: {score:.2%}",
                ))

        return results

    def find_similar(self, preset: "VoicePreset", top_k: int = 5) -> list[SearchResult]:
        """
        Find presets similar to a given preset.

        Args:
            preset: Reference preset.
            top_k: Number of results.

        Returns:
            List of similar presets (excluding the input preset).
        """
        results = self.search(preset.get_search_text(), top_k=top_k + 1)

        # Filter out the input preset if present
        return [r for r in results if r.preset.id != preset.id][:top_k]

    @staticmethod
    def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and corpus vectors."""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)

        # Dot product gives cosine similarity for normalized vectors
        return np.dot(corpus_norm, query_norm)


class FallbackSearch:
    """
    Fallback search using keyword matching.

    Used when sentence-transformers is not available.
    """

    def __init__(self):
        self._presets: list["VoicePreset"] = []
        self._indexed = False

    def index(self, presets: list["VoicePreset"]) -> "FallbackSearch":
        """Index presets for keyword search."""
        self._presets = list(presets)
        self._indexed = True
        return self

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search using keyword matching.

        Scores based on number of query terms found in preset text.
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_terms = set(query.lower().split())
        results = []

        for preset in self._presets:
            search_text = preset.get_search_text().lower()
            search_words = set(search_text.split())

            # Count matching terms
            matches = query_terms & search_words
            if matches:
                score = len(matches) / len(query_terms)
                results.append(SearchResult(
                    preset=preset,
                    score=score,
                    match_reason=f"Matched terms: {', '.join(sorted(matches))}",
                ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


def create_search_engine(use_semantic: bool = True) -> SemanticSearch | FallbackSearch:
    """
    Create a search engine, with fallback if semantic search unavailable.

    Args:
        use_semantic: Whether to try semantic search first.

    Returns:
        SemanticSearch if available, else FallbackSearch.
    """
    if use_semantic:
        try:
            return SemanticSearch()
        except ImportError:
            logger.info("Semantic search unavailable, using keyword fallback")

    return FallbackSearch()
