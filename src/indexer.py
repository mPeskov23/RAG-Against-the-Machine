from models import MinimalSource


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