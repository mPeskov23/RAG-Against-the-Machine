"""Lexical BM25 indexer module.

Implements inverted index, code-aware tokenization
(CamelCase, snake_case, path tokens), and BM25 ranking for sub-second
retrieval over code and documentation.
"""

from collections import Counter
import math
from pathlib import Path
import pickle
import re
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
from .chunking import chunk, read_file
from .models import MinimalSource


STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
}


def split_identifier(identifier: str) -> List[str]:
    """Split identifier into subwords by underscores and CamelCase
    boundaries."""
    results: List[str] = []
    lowered = identifier.lower()
    results.append(lowered)

    if "_" in identifier:
        for part in identifier.split("_"):
            part_clean = part.strip().lower()
            if len(part_clean) > 1 and part_clean not in STOPWORDS:
                results.append(part_clean)

    camel_parts = re.findall(
        r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[0-9]+", identifier
    )
    if len(camel_parts) > 1:
        for part in camel_parts:
            part_clean = part.strip().lower()
            if len(part_clean) > 1 and part_clean not in STOPWORDS:
                results.append(part_clean)

    return results


def tokenize(text: str, file_path: str = "") -> List[str]:
    """Extract tokens from text and file path with code identifier
    decomposition."""
    tokens: List[str] = []

    # File path tokens (boosted for matching module/file names)
    if file_path:
        path_parts = re.findall(r"[a-zA-Z0-9_]+", file_path)
        for part in path_parts:
            if part.lower() in {"data", "raw", "vllm", "py", "md", "src"}:
                continue
            subparts = split_identifier(part)
            for sp in subparts:
                if len(sp) > 1 and sp not in STOPWORDS:
                    # Double-weight path tokens to reinforce document identity
                    tokens.extend([sp, sp])

    # Body tokens
    words = re.findall(r"[a-zA-Z0-9_]+", text)
    for word in words:
        if len(word) <= 1:
            continue
        subparts = split_identifier(word)
        for sp in subparts:
            if len(sp) > 1 and sp not in STOPWORDS:
                tokens.append(sp)

    return tokens


def get_flat_chunks(
    dir_path: str, max_chunk_size: int = 2000
) -> List[MinimalSource]:
    """Recursively collect chunks from all Python and Markdown files
    under dir_path."""
    chunk_list: List[MinimalSource] = []
    base = Path(dir_path)
    all_files = [
        p for p in base.rglob("*")
        if p.is_file() and p.suffix in [".py", ".md"]
    ]

    for file_path in tqdm(all_files, desc="Chunking files", unit="file"):
        chunk_list.extend(chunk(str(file_path), max_chunk_size=max_chunk_size))

    return chunk_list


class BM25Indexer:
    """Okapi BM25 indexer with inverted index and fast sparse scoring."""

    def __init__(self, corpus: Optional[List[MinimalSource]] = None) -> None:
        """Initialize BM25 indexer with optional corpus."""
        self.corpus: List[MinimalSource] = corpus or []
        self.doc_count: int = len(self.corpus)
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.idf: Dict[str, float] = {}
        self.k1: float = 1.2
        self.b: float = 0.75

    def create_inverted_index(self) -> None:
        """Build term frequency dictionary and inverted index
        for all corpus documents."""
        file_cache: Dict[str, str] = {}
        self.doc_lengths = []
        self.inverted_index = {}

        for doc_id, entry in enumerate(
            tqdm(self.corpus, desc="Indexing chunks", unit="chunk")
        ):
            name = entry.file_path
            start = entry.first_character_index
            end = entry.last_character_index

            if name not in file_cache:
                file_cache[name] = read_file(name)

            file_text = file_cache[name]
            text_chunk = file_text[start:end]
            tokens = tokenize(text_chunk, file_path=name)

            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)

            tf_counter = Counter(tokens)
            for token, freq in tf_counter.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = freq

    def calculate_idf(self) -> None:
        """Calculate Okapi BM25 Inverse Document Frequency
        for all indexed terms."""
        self.idf = {}
        n = self.doc_count
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def fit(self) -> None:
        """Construct full inverted index and compute global statistics."""
        self.doc_count = len(self.corpus)
        self.create_inverted_index()
        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_lengths) / self.doc_count
        else:
            self.avg_doc_len = 0.0
        self.calculate_idf()

    def score_query(
        self,
        query: str,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Score all documents against query tokens using BM25 formula."""
        k1_val = k1 if k1 is not None else self.k1
        b_val = b if b is not None else self.b

        query_tokens = tokenize(query)
        if not query_tokens or self.doc_count == 0 or self.avg_doc_len == 0.0:
            return []

        scores: Dict[int, float] = {}
        query_freqs = Counter(query_tokens)

        for token, q_tf in query_freqs.items():
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            if idf <= 0.0:
                continue

            postings = self.inverted_index[token]
            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths[doc_id]
                len_norm = 1.0 - b_val + b_val * (doc_len / self.avg_doc_len)
                denom = tf + k1_val * len_norm
                term_score = idf * (tf * (k1_val + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def save(
        self, file_path: str | Path = "data/processed/bm25_indexer.pkl"
    ) -> Path:
        """Persist indexer instance to disk via pickle."""
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "bm25_indexer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(
        cls, file_path: str | Path = "data/processed/bm25_indexer.pkl"
    ) -> "BM25Indexer":
        """Load pickled BM25Indexer instance from disk."""
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "bm25_indexer.pkl"
        with open(path, "rb") as f:
            indexer: BM25Indexer = pickle.load(f)
        return indexer
