from .models import MinimalSource
from collections import Counter
import re


def tokenize(text: str) -> list[str]:
    text_lower = text.lower()
    retlist = []
    raw_list = re.findall(r"[a-zA-Z0-9_+]", text_lower)
    for item in raw_list:
        if len(item) > 1:
            if '_' in item:
                for lower_item in item.split('_'):
                    retlist.append(lower_item)
            retlist.append(item)
    return retlist


def get_tf_counter(tokens: list[str]):
    return Counter(tokens)


class BM25Indexer:
    def __init__(self,
                 corpus: list[MinimalSource],
                 doc_lengths: list[int],
                 avg_doc_len: float,
                 doc_count: int,
                 inverted_index: dict[str, dict[int, int]],
                 idf: dict[str, float]) -> None:
        self.corpus = corpus
        self.doc_lengths = doc_lengths
        self.avg_doc_len = avg_doc_len
        self.doc_count = doc_count
        self.inverted_index = inverted_index
        self.idf = idf