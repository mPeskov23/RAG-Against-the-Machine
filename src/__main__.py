from .indexer import BM25Indexer, get_flat_chunks


MAX_CHUNK_SIZE = 2000

if __name__ == "__main__":
    sources = get_flat_chunks("./data/raw", MAX_CHUNK_SIZE)
    indexer = BM25Indexer(sources)
    indexer.fit()
    print(indexer.score_query("How to mount an openAI server?", top_k=5))
    print(indexer.corpus[15265])