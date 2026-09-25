"""Caching module for index and query results.

Cache index objects in memory to avoid cold starts and cache query results
to accelerate repeated queries.
"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional
from .embeddings import VectorIndexer
from .hybrid import HybridRetriever
from .indexer import BM25Indexer
from .models import MinimalSource


DEFAULT_CACHE_FILE = "data/processed/query_cache.json"


class QueryCache:
    """LRU cache with optional JSON disk persistence
    for search and answer results."""

    def __init__(
        self,
        capacity: int = 500,
        cache_file: str | Path = DEFAULT_CACHE_FILE,
    ) -> None:
        """Initialize QueryCache with maximum capacity
        and disk persistence path."""
        self.capacity = capacity
        self.cache_file = Path(cache_file)
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self._load()

    def _make_key(self, query: str, k: int, **kwargs: Any) -> str:
        """Create deterministic hashed cache key from query parameters."""
        raw = f"{query.strip().lower()}:{k}:{sorted(kwargs.items())}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, k: int, **kwargs: Any) -> Optional[Any]:
        """Retrieve cached result if present, updating LRU order."""
        key = self._make_key(query, k, **kwargs)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, query: str, k: int, value: Any, **kwargs: Any) -> None:
        """Store result in cache,
        evicting oldest item if capacity is exceeded."""
        key = self._make_key(query, k, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        self.save()

    def save(self) -> None:
        """Serialize cache entries to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        """Load persisted cache entries from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for k, v in entries.items():
                    self._cache[k] = v
            except Exception:
                pass


class IndexCache:
    """In-memory singleton cache for indexer instances
    to accelerate cold start."""

    _bm25_instance: Optional[BM25Indexer] = None
    _vector_instance: Optional[VectorIndexer] = None
    _query_cache: Optional[QueryCache] = None

    @classmethod
    def get_bm25(
        cls, path: str | Path = "data/processed/bm25_indexer.pkl"
    ) -> BM25Indexer:
        """Retrieve or load cached BM25Indexer instance."""
        if cls._bm25_instance is None:
            cls._bm25_instance = BM25Indexer.load(path)
        return cls._bm25_instance

    @classmethod
    def get_vector(
        cls, path: str | Path = "data/processed/vector_indexer.pkl"
    ) -> Optional[VectorIndexer]:
        """Retrieve or load cached VectorIndexer instance if available."""
        if cls._vector_instance is None:
            p = Path(path)
            if p.exists():
                try:
                    cls._vector_instance = VectorIndexer.load(p)
                except Exception:
                    cls._vector_instance = None
        return cls._vector_instance

    @classmethod
    def get_query_cache(cls) -> QueryCache:
        """Retrieve or initialize singleton QueryCache instance."""
        if cls._query_cache is None:
            cls._query_cache = QueryCache()
        return cls._query_cache

    @classmethod
    def cached_search(
        cls,
        indexer: BM25Indexer,
        query: str,
        k: int = 10,
        hybrid_retriever: Optional[HybridRetriever] = None,
        doc_type: Optional[str] = None,
    ) -> List[MinimalSource]:
        """Perform search with query cache acceleration."""
        cache = cls.get_query_cache()
        use_hybrid = hybrid_retriever is not None
        cached = cache.get(query, k, hybrid=use_hybrid, doc_type=doc_type)
        if cached is not None and isinstance(cached, list):
            cached_sources: List[MinimalSource] = []
            for item in cached:
                if isinstance(item, dict):
                    cached_sources.append(
                        MinimalSource(
                            file_path=str(item.get("file_path", "")),
                            first_character_index=int(
                                item.get("first_character_index", 0)
                            ),
                            last_character_index=int(
                                item.get("last_character_index", 0)
                            ),
                        )
                    )
            return cached_sources

        if hybrid_retriever is not None:
            results = hybrid_retriever.retrieve(query, top_k=k)
        else:
            scored = indexer.score_query(query, top_k=k, doc_type=doc_type)
            results = [indexer.corpus[doc_id] for doc_id, _ in scored]

        cache.set(
            query,
            k,
            [r.model_dump() for r in results],
            hybrid=use_hybrid,
            doc_type=doc_type,
        )
        return results
