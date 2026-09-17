"""Hybrid retrieval combining lexical BM25 and semantic vector rankings.

Bonus 2: Reciprocal Rank Fusion (RRF) combining sparse BM25 and dense embeddings.
"""

from typing import Dict, List, Optional, Tuple
from .embeddings import VectorIndexer
from .indexer import BM25Indexer
from .models import MinimalSource


class HybridRetriever:
    """Combines BM25 and vector embeddings rankings using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        bm25_indexer: BM25Indexer,
        vector_indexer: Optional[VectorIndexer] = None,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        semantic_weight: float = 0.8,
    ) -> None:
        """Initialize hybrid retriever with lexical and optional semantic indexers."""
        self.bm25_indexer = bm25_indexer
        self.vector_indexer = vector_indexer
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
    ) -> List[MinimalSource]:
        """Retrieve top-k sources using RRF hybrid fusion or standalone BM25."""
        if top_k <= 0 or not query.strip():
            return []

        bm25_results: List[Tuple[int, float]] = self.bm25_indexer.score_query(
            query, top_k=top_k * 3
        )

        if not use_hybrid or self.vector_indexer is None:
            return [self.bm25_indexer.corpus[doc_id] for doc_id, _ in bm25_results[:top_k]]

        vec_results: List[Tuple[int, float]] = self.vector_indexer.score_query(
            query, top_k=top_k * 3
        )

        # Reciprocal Rank Fusion
        rrf_scores: Dict[Tuple[str, int, int], float] = {}
        source_map: Dict[Tuple[str, int, int], MinimalSource] = {}

        for rank, (doc_id, _) in enumerate(bm25_results):
            src = self.bm25_indexer.corpus[doc_id]
            key = (src.file_path, src.first_character_index, src.last_character_index)
            source_map[key] = src
            score = self.lexical_weight / (self.rrf_k + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

        for rank, (doc_id, _) in enumerate(vec_results):
            src = self.vector_indexer.corpus[doc_id]
            key = (src.file_path, src.first_character_index, src.last_character_index)
            source_map[key] = src
            score = self.semantic_weight / (self.rrf_k + rank + 1)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True
        )

        return [source_map[k] for k in sorted_keys[:top_k]]
