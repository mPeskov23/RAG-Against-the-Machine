"""Command-Line Interface for RAG Against the Machine.

Provides full Python Fire CLI.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from .api import start_server
from .caching import IndexCache
from .chunking import chunk
from .embeddings import VectorIndexer
from .generator import AnswerGenerator
from .hybrid import HybridRetriever
from .incremental import IncrementalIndexManager
from .indexer import BM25Indexer, get_flat_chunks
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


def compute_iou(
    s1_start: int, s1_end: int, s2_start: int, s2_end: int
) -> float:
    """Compute Intersection over Union (IoU) between two character spans."""
    inter_start = max(s1_start, s2_start)
    inter_end = min(s1_end, s2_end)
    inter = max(0, inter_end - inter_start)
    union = (s1_end - s1_start) + (s2_end - s2_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


class RagCLI:
    """Command Line Interface for indexing, searching, answering,
    and evaluating."""

    def index(
        self,
        corpus_dir: str = "data/raw",
        max_chunk_size: int = 2000,
        processed_dir: str = "data/processed",
        incremental: bool = False,
        with_embeddings: bool = False,
    ) -> None:
        """Ingest corpus and construct search indices under data/processed/."""
        try:
            p_dir = Path(processed_dir)
            p_dir.mkdir(parents=True, exist_ok=True)
            bm25_path = p_dir / "bm25_indexer.pkl"

            if incremental and bm25_path.exists():
                print(f"Performing incremental indexing on {corpus_dir}...")
                indexer = BM25Indexer.load(bm25_path)
                inc_mgr = IncrementalIndexManager()
                added, modified, deleted = inc_mgr.update_index(
                    indexer, corpus_dir, max_chunk_size=max_chunk_size
                )
                indexer.save(bm25_path)
                print(
                    f"Incremental indexing complete: {added} added, "
                    f"{modified} modified, {deleted} deleted. "
                    f"Total chunks: {len(indexer.corpus)}."
                )
                return

            print(
                f"Reading and chunking corpus from {corpus_dir} "
                f"(max_chunk_size={max_chunk_size})..."
            )
            chunks = get_flat_chunks(corpus_dir, max_chunk_size=max_chunk_size)
            if not chunks:
                print(
                    "Warning: No valid files found to index under "
                    f"{corpus_dir}."
                )
                return

            print(f"Collected {len(chunks)} chunks. Fitting BM25 indexer...")
            indexer = BM25Indexer(chunks)
            indexer.fit()
            indexer.save(bm25_path)

            # Record state for incremental indexing
            inc_mgr = IncrementalIndexManager()
            for p in Path(corpus_dir).rglob("*"):
                if p.is_file() and p.suffix in [".py", ".md"]:
                    file_chunks = chunk(str(p), max_chunk_size=max_chunk_size)
                    inc_mgr.registry[str(p)] = {
                        "mtime": p.stat().st_mtime,
                        "hash": "",
                        "chunks": len(file_chunks),
                    }
            inc_mgr.save_registry()

            if with_embeddings:
                print(
                    "Building semantic vector index with all-MiniLM-L6-v2..."
                )
                vec_indexer = VectorIndexer(chunks)
                vec_indexer.fit(batch_size=128)
                vec_indexer.save(p_dir / "vector_indexer.pkl")
                print("Semantic vector index saved!")

            print(f"Ingestion complete! Indices saved under {processed_dir}/")
        except Exception as exc:
            print(f"Error during indexing: {exc}", file=sys.stderr)

    def search(
        self,
        query: str = "",
        k: int = 10,
        index_path: str = "data/processed/bm25_indexer.pkl",
        hybrid: bool = False,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k sources for a single natural language query."""
        try:
            if not query or not query.strip() or k <= 0:
                print(
                    "Empty query or k <= 0 provided. Returning empty results."
                )
                return []

            p = Path(index_path)
            if not p.exists():
                print(
                    f"Error: Index file not found at {index_path}. "
                    "Run index first.",
                    file=sys.stderr,
                )
                return []

            indexer = IndexCache.get_bm25(p)
            vec_indexer = IndexCache.get_vector() if hybrid else None
            retriever = None
            if hybrid and vec_indexer is not None:
                retriever = HybridRetriever(
                    bm25_indexer=indexer, vector_indexer=vec_indexer
                )

            sources = IndexCache.cached_search(
                indexer,
                query,
                k=k,
                hybrid_retriever=retriever,
                doc_type=doc_type,
            )

            for i, src in enumerate(sources, 1):
                start_c = src.first_character_index
                end_c = src.last_character_index
                span = f"({start_c}-{end_c})"
                print(f"[{i}] {src.file_path} {span}")

            return [src.model_dump() for src in sources]
        except Exception as exc:
            print(f"Error executing search: {exc}", file=sys.stderr)
            return []

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_path: str = "data/processed/bm25_indexer.pkl",
        hybrid: bool = False,
        doc_type: Optional[str] = None,
    ) -> Optional[StudentSearchResults]:
        """Run search across all questions in a dataset and save
        StudentSearchResults JSON."""
        try:
            if not dataset_path:
                print("Error: --dataset_path is required.", file=sys.stderr)
                return None

            d_path = Path(dataset_path)
            if not d_path.exists():
                print(
                    f"Error: Dataset file not found at {dataset_path}",
                    file=sys.stderr,
                )
                return None

            try:
                with open(d_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as exc:
                print(f"Error reading dataset JSON: {exc}", file=sys.stderr)
                return None

            rag_dataset = RagDataset.model_validate(raw_data)
            indexer = IndexCache.get_bm25(index_path)
            vec_indexer = IndexCache.get_vector() if hybrid else None
            retriever = None
            if hybrid and vec_indexer is not None:
                retriever = HybridRetriever(
                    bm25_indexer=indexer, vector_indexer=vec_indexer
                )

            resolved_doc_type = doc_type
            if resolved_doc_type is None:
                stem_name = d_path.stem.lower()
                if "doc" in stem_name:
                    resolved_doc_type = "docs"
                elif "code" in stem_name:
                    resolved_doc_type = "code"

            results: List[MinimalSearchResults] = []
            questions = rag_dataset.rag_questions

            for q in tqdm(questions, desc="Searching questions", unit="query"):
                q_id = q.question_id
                q_text = q.question
                if not q_text.strip() or k <= 0:
                    retrieved_sources: List[MinimalSource] = []
                else:
                    retrieved_sources = IndexCache.cached_search(
                        indexer,
                        q_text,
                        k=k,
                        hybrid_retriever=retriever,
                        doc_type=resolved_doc_type,
                    )

                results.append(
                    MinimalSearchResults(
                        question_id=q_id,
                        question=q_text,
                        retrieved_sources=retrieved_sources,
                    )
                )

            output = StudentSearchResults(search_results=results, k=k)

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_file = save_dir / d_path.name

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(output.model_dump_json(indent=2))

            print(f"Saved student_search_results to {out_file}")
            return output
        except Exception as exc:
            print(f"Error in search_dataset: {exc}", file=sys.stderr)
            return None

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_path: str = "data/processed/bm25_indexer.pkl",
        hybrid: bool = False,
    ) -> str:
        """Answer a single query using retrieved context."""
        try:
            if not query or not query.strip():
                print("Empty query provided. Cannot generate answer.")
                return ""

            p = Path(index_path)
            if not p.exists():
                print(
                    f"Error: Index file not found at {index_path}. "
                    "Please run 'uv run python3 -m src index' first.",
                    file=sys.stderr,
                )
                return ""

            indexer = IndexCache.get_bm25(p)
            vec_indexer = IndexCache.get_vector() if hybrid else None
            retriever = None
            if hybrid and vec_indexer is not None:
                retriever = HybridRetriever(
                    bm25_indexer=indexer, vector_indexer=vec_indexer
                )

            sources = IndexCache.cached_search(
                indexer, query, k=k, hybrid_retriever=retriever
            )
            generator = AnswerGenerator.get_instance()
            answer_text = generator.generate_answer(query, sources)

            return answer_text
        except Exception as exc:
            print(f"Error generating answer: {exc}", file=sys.stderr)
            return ""

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
    ) -> Optional[StudentSearchResultsAndAnswer]:
        """Generate grounded answers for search results and save
        StudentSearchResultsAndAnswer."""
        try:
            if not student_search_results_path:
                print(
                    "Error: --student_search_results_path is required.",
                    file=sys.stderr,
                )
                return None

            in_path = Path(student_search_results_path)
            if not in_path.exists():
                print(
                    f"Error: Search results file not found at {in_path}",
                    file=sys.stderr,
                )
                return None

            try:
                with open(in_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as exc:
                print(
                    f"Error reading search results JSON: {exc}",
                    file=sys.stderr,
                )
                return None

            search_output = StudentSearchResults.model_validate(raw_data)
            generator = AnswerGenerator.get_instance()

            answered_results: List[MinimalAnswer] = []
            total = len(search_output.search_results)
            print(f"Loaded {total} questions ...")

            for item in tqdm(
                search_output.search_results,
                desc="Generating answers",
                unit="q",
            ):
                ans = generator.generate_answer(
                    item.question, item.retrieved_sources
                )
                answered_results.append(
                    MinimalAnswer(
                        question_id=item.question_id,
                        question=item.question,
                        retrieved_sources=item.retrieved_sources,
                        answer=ans,
                    )
                )

            final_output = StudentSearchResultsAndAnswer(
                search_results=answered_results,
                k=search_output.k,
            )

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_file = save_dir / in_path.name

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))

            print(f"Processed {len(answered_results)} of {total} questions")
            print(f"Saved student_search_results_and_answer to {out_file}")
            return final_output
        except Exception as exc:
            print(f"Error in answer_dataset: {exc}", file=sys.stderr)
            return None

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
        k: int = 10,
        max_context_length: int = 2000,
    ) -> Dict[str, float]:
        """Calculate Recall@1, 3, 5, 10 against ground-truth dataset
        for local testing."""
        try:
            if not student_search_results_path or not dataset_path:
                print(
                    "Error: Both --student_search_results_path and "
                    "--dataset_path are required."
                )
                return {}

            res_path = Path(student_search_results_path)
            gt_path = Path(dataset_path)

            if not res_path.exists() or not gt_path.exists():
                print(
                    "Error: One or more input paths do not exist.",
                    file=sys.stderr,
                )
                return {}

            with open(res_path, "r", encoding="utf-8") as f:
                student_results = StudentSearchResults.model_validate(
                    json.load(f)
                )

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_dataset = RagDataset.model_validate(json.load(f))

            student_map: Dict[str, List[MinimalSource]] = {
                item.question_id: item.retrieved_sources
                for item in student_results.search_results
            }

            recall_k_vals = [1, 3, 5, 10]
            hits: Dict[int, int] = {val: 0 for val in recall_k_vals}
            total_evaluated = 0

            for q in gt_dataset.rag_questions:
                if not isinstance(q, AnsweredQuestion) or not q.sources:
                    continue

                total_evaluated += 1
                retrieved = student_map.get(q.question_id, [])

                # Filter out any sources exceeding max_context_length
                valid_retrieved = [
                    src for src in retrieved
                    if (
                        src.last_character_index - src.first_character_index
                    ) <= max_context_length
                ]

                for val in recall_k_vals:
                    top_sources = valid_retrieved[:val]
                    found = False
                    for gt_src in q.sources:
                        for r_src in top_sources:
                            if r_src.file_path == gt_src.file_path:
                                iou = compute_iou(
                                    gt_src.first_character_index,
                                    gt_src.last_character_index,
                                    r_src.first_character_index,
                                    r_src.last_character_index,
                                )
                                if iou >= 0.05:
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        hits[val] += 1

            results: Dict[str, float] = {}
            if total_evaluated > 0:
                print("\nEvaluation Results")
                print("========================================")
                print(f"Questions evaluated: {total_evaluated}")
                for val in recall_k_vals:
                    score = hits[val] / total_evaluated
                    results[f"Recall@{val}"] = round(score, 4)
                    print(f"Recall@{val}: {score:.4f}")
            return results
        except Exception as exc:
            print(f"Error during evaluation: {exc}", file=sys.stderr)
            return {}

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start local HTTP API server for Bonus 5."""
        try:
            print(
                f"Starting local HTTP API server on http://{host}:{port} ..."
            )
            start_server(host=host, port=port)
        except Exception as exc:
            print(f"Error running server: {exc}", file=sys.stderr)
