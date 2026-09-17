"""Semantic vector indexer module using lightweight all-MiniLM-L6-v2.

Bonus 1: Semantic embeddings vector index next to lexical BM25 index.
"""

from pathlib import Path
import pickle
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .chunking import read_file
from .models import MinimalSource


MODEL_NAME = "all-MiniLM-L6-v2"


class VectorIndexer:
    """Dense vector indexer using SentenceTransformer for semantic retrieval."""

    def __init__(
        self,
        corpus: Optional[List[MinimalSource]] = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        """Initialize VectorIndexer with optional corpus and CPU sentence transformer."""
        self.corpus: List[MinimalSource] = corpus or []
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load SentenceTransformer model on CPU."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def fit(self, batch_size: int = 128, max_docs: Optional[int] = None) -> None:
        """Encode all corpus chunks into normalized dense vectors."""
        if not self.corpus:
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            return

        corpus_subset = self.corpus[:max_docs] if max_docs else self.corpus
        file_cache: dict[str, str] = {}
        texts: List[str] = []

        for item in tqdm(corpus_subset, desc="Preparing semantic chunks", unit="chunk"):
            if item.file_path not in file_cache:
                file_cache[item.file_path] = read_file(item.file_path)
            content = file_cache[item.file_path]
            snippet = content[item.first_character_index:item.last_character_index].strip()
            # Include file path header for semantic context
            texts.append(f"{item.file_path}\n{snippet[:800]}")

        self.corpus = corpus_subset
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def score_query(
        self, query: str, top_k: Optional[int] = 10
    ) -> List[Tuple[int, float]]:
        """Score all documents against query using cosine similarity over normalized vectors."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Dot product of normalized vectors is cosine similarity
        similarities = np.dot(self.embeddings, query_vec)

        if top_k is not None and top_k < len(similarities):
            # Fast partial partition
            top_indices = np.argpartition(-similarities, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
        else:
            top_indices = np.argsort(-similarities)
            if top_k is not None:
                top_indices = top_indices[:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def save(
        self, file_path: str | Path = "data/processed/vector_indexer.pkl"
    ) -> Path:
        """Persist vector index and corpus metadata to disk."""
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "vector_indexer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Avoid pickling the PyTorch model itself
        saved_model = self._model
        self._model = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self._model = saved_model
        return path

    @classmethod
    def load(
        cls, file_path: str | Path = "data/processed/vector_indexer.pkl"
    ) -> "VectorIndexer":
        """Load pickled VectorIndexer from disk."""
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "vector_indexer.pkl"
        with open(path, "rb") as f:
            indexer: VectorIndexer = pickle.load(f)
        return indexer
