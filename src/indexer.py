from .models import MinimalSource
from .chunking import chunk, read_file
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import re
import math


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
        for doc_id, entry in enumerate(tqdm(self.corpus, desc="Creating Index")):
            name = entry.file_path
            start = entry.first_character_index
            end = entry.last_character_index
            if name not in file_cache:
                file_cache[name] = read_file(name)
            file_text = file_cache[name]
            chunk = file_text[start:end]
            tokens = tokenize(chunk)
            self.doc_lengths.append(len(tokens))
            for token in tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = {doc_id: 1}
                else:
                    self.inverted_index[token][doc_id] += self.inverted_index[token].get(doc_id, 0) + 1

    def calculate_inv_doc_fr(self) -> None:
        for key in tqdm(self.inverted_index.keys(), dec="Calculating idf"):
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
