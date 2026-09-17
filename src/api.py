"""Local HTTP API module using FastAPI.

Expose querying index and answering questions over a local HTTP API.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException
import uvicorn
from .caching import IndexCache
from .generator import AnswerGenerator
from .hybrid import HybridRetriever
from .models import (
    AnswerRequest,
    HealthResponse,
    MinimalAnswer,
    MinimalSearchResults,
    SearchRequest,
)


app = FastAPI(
    title="mpeskov Against the Machine API",
    description="Local HTTP API for codebase retrieval and grounded question answering",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check API health and index readiness."""
    try:
        bm25 = IndexCache.get_bm25()
        doc_count = bm25.doc_count
        lexical = True
    except Exception:
        doc_count = 0
        lexical = False

    vec = IndexCache.get_vector()
    semantic = vec is not None and vec.embeddings is not None and len(vec.embeddings) > 0

    return HealthResponse(
        status="ok" if lexical else "no_index",
        indexed_documents=doc_count,
        lexical_index=lexical,
        semantic_index=semantic,
    )


@app.post("/search", response_model=MinimalSearchResults)
def search_endpoint(request: SearchRequest) -> MinimalSearchResults:
    """Retrieve top-k relevant source locations for a question."""
    try:
        bm25 = IndexCache.get_bm25()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index not available: {exc}")

    vec = IndexCache.get_vector() if request.hybrid else None
    retriever: Optional[HybridRetriever] = None
    if request.hybrid and vec is not None:
        retriever = HybridRetriever(bm25_indexer=bm25, vector_indexer=vec)

    sources = IndexCache.cached_search(
        indexer=bm25,
        query=request.query,
        k=request.k,
        hybrid_retriever=retriever,
    )

    return MinimalSearchResults(
        question_id="api_search",
        question=request.query,
        retrieved_sources=sources,
    )


@app.post("/answer", response_model=MinimalAnswer)
def answer_endpoint(request: AnswerRequest) -> MinimalAnswer:
    """Answer question grounded in retrieved codebase snippets."""
    try:
        bm25 = IndexCache.get_bm25()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index not available: {exc}")

    vec = IndexCache.get_vector() if request.hybrid else None
    retriever: Optional[HybridRetriever] = None
    if request.hybrid and vec is not None:
        retriever = HybridRetriever(bm25_indexer=bm25, vector_indexer=vec)

    sources = IndexCache.cached_search(
        indexer=bm25,
        query=request.query,
        k=request.k,
        hybrid_retriever=retriever,
    )

    gen = AnswerGenerator.get_instance()
    answer = gen.generate_answer(request.query, sources)

    return MinimalAnswer(
        question_id="api_answer",
        question=request.query,
        retrieved_sources=sources,
        answer=answer,
    )


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run uvicorn server for local HTTP API."""
    uvicorn.run(app, host=host, port=port)
