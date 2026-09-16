from .models import MinimalSource
from .chunking import chunk
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import re


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

    def fit(self) -> None:
        pass
