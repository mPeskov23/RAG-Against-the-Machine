import math
import pickle
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from .chunking import chunk, read_file
from .models import MinimalSource


def tokenize(text: str) -> list[str]:
    text_lower = text.lower()
    retlist = []
    raw_list = re.findall(r"[a-zA-Z0-9_]+", text_lower)
    for item in raw_list:
        if len(item) > 1:
            if '_' in item:
                for lower_item in item.split('_'):
                    if len(lower_item) > 1:
                        retlist.append(lower_item)
            retlist.append(item)
    return retlist


def get_tf_counter(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def get_flat_chunks(dir_path: str,
                    max_chunk_size: int = 2000) -> list[MinimalSource]:
    retlist: list[MinimalSource] = []
    for path in tqdm(Path(dir_path).rglob("*"), desc="Chunking"):
        if path.is_file() and path.suffix in [".py", ".md"]:
            retlist.extend(chunk(str(path), max_chunk_size=max_chunk_size))
    return retlist


class BM25Indexer:
    def __init__(self, corpus: list[MinimalSource]) -> None:
        self.corpus = corpus
        self.doc_count: int = len(corpus)
        self.doc_lengths: list[int] = []
        self.avg_doc_len: float = 0.0
        self.inverted_index: dict[str, dict[int, int]] = {}
        self.idf: dict[str, float] = {}

    def create_inverted_index(self) -> None:
        file_cache: dict[str, str] = {}
        for doc_id, entry in enumerate(
            tqdm(self.corpus, desc="Creating Index", unit="doc")
        ):
            name = entry.file_path
            start = entry.first_character_index
            end = entry.last_character_index
            if name not in file_cache:
                file_cache[name] = read_file(name)
            file_text = file_cache[name]
            text_chunk = file_text[start:end]
            tokens = tokenize(text_chunk)
            self.doc_lengths.append(len(tokens))
            tf_counter = get_tf_counter(tokens)
            for token, tf in tf_counter.items():
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}
                self.inverted_index[token][doc_id] = tf

    def calculate_inv_doc_fr(self) -> None:
        for key in tqdm(self.inverted_index.keys(), desc="Calculating idf"):
            if key not in self.idf:
                self.idf[key] = 0.0
            q = len(self.inverted_index[key])
            n = self.doc_count
            self.idf[key] = math.log(1 + (n - q + 0.5) / (q + 0.5))

    def fit(self) -> None:
        self.create_inverted_index()
        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_lengths) / self.doc_count
        self.calculate_inv_doc_fr()

    def save(
        self, file_path: str | Path = "data/processed/bm25_indexer.pkl"
    ) -> Path:
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "bm25_indexer.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(
        cls, file_path: str | Path = "data/processed/bm25_indexer.pkl"
    ) -> "BM25Indexer":
        path = Path(file_path)
        if path.is_dir() or not path.suffix:
            path = path / "bm25_indexer.pkl"
        with open(path, "rb") as f:
            indexer: BM25Indexer = pickle.load(f)
        return indexer

    def score_query(self,
                    query: str,
                    k: float = 1.5,
                    b: float = 0.75,
                    top_k: int | None = None) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: dict[int, float] = dict()
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf[token]
            docs = self.inverted_index[token]
            for doc_id, freq in docs.items():
                doc_len = self.doc_lengths[doc_id]
                length_norm = 1 - b + b * doc_len / self.avg_doc_len
                numerator = freq * (k + 1)
                denominator = freq * (k + length_norm)
                term_score = idf * numerator / denominator
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score
        results = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        if top_k is not None:
            results = results[:top_k]
        return results
