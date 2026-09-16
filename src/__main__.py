from pathlib import Path
from .indexer import BM25Indexer, get_flat_chunks


MAX_CHUNK_SIZE = 2000
PROCESSED_DIR = Path("data/processed")
INDEX_PATH = PROCESSED_DIR / "bm25_indexer.pkl"

if __name__ == "__main__":
    if INDEX_PATH.exists():
        print(f"Loading cached index from {INDEX_PATH}...")
        indexer = BM25Indexer.load(INDEX_PATH)
    else:
        print("Index not found. Building from raw data...")
        sources = get_flat_chunks("./data/raw", MAX_CHUNK_SIZE)
        indexer = BM25Indexer(sources)
        indexer.fit()
        indexer.save(INDEX_PATH)
        print(f"Index successfully fitted and saved to {INDEX_PATH}")

    results = indexer.score_query("How to mount an openAI server?", top_k=5)
    print(results)
    if results:
        top_doc_id = results[0][0]
        print(f"Top result (doc_id={top_doc_id}):")
        print(indexer.corpus[top_doc_id])
