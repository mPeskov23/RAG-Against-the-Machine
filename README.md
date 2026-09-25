*This project has been created as part of the 42 curriculum by mpeskov.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is a high-performance Retrieval-Augmented Generation (RAG) system engineered to answer complex technical questions grounded in the vLLM codebase. 

Instead of relying solely on parametric weights that are frozen in time, the system ingests large-scale source repositories into a searchable lexical and semantic index, retrieves relevant code and documentation snippets with high precision and recall, and produces faithful, grounded natural-language answers using `Qwen/Qwen3-0.6B`.

The project features:
- Dual-mode semantic document chunking (AST-based for Python and hierarchical header-based for Markdown).
- High-throughput BM25 indexer with code-aware tokenization (CamelCase, snake_case, file-path weighting).
- Grounded answer generation using `Qwen/Qwen3-0.6B` on CPU with prompt engineering and context budget control.
- All 5 bonus extensions implemented: Semantic vector embeddings, Hybrid RRF retrieval, Incremental indexing, In-memory and disk query caching, and a local REST HTTP API.

---

## System Architecture

The pipeline consists of modular components exchanging validated Pydantic data structures:

```
                  +----------------------------------------+
                  |         Raw Codebase (vLLM)            |
                  |     (Python files & Markdown docs)     |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |            Chunking Engine             |
                  |  - Python AST (preamble, classes, fns) |
                  |  - Markdown (header hierarchy, \n\n)   |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |            Indexing Engine             |
                  |  - BM25 Inverted Index (Lexical)       |
                  |  - all-MiniLM-L6-v2 Vectors (Dense)    |
                  |  - Incremental Manager (mtime/hash)    |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |           Retrieval Engine             |
                  |  - Path-weighted Lexical BM25          |
                  |  - Dense Cosine Similarity Search      |
                  |  - Reciprocal Rank Fusion (RRF) Hybrid |
                  |  - Multi-level Query & Index Caching   |
                  +-------------------+--------------------+
                                      |
                                      v
                  +----------------------------------------+
                  |          Generation Engine             |
                  |  - Context extraction & token budget   |
                  |  - Qwen/Qwen3-0.6B causal LM on CPU    |
                  +-------------------+--------------------+
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
               CLI (Python Fire)           Local HTTP API (FastAPI)
```

1. **Chunking Engine (`src/chunking.py`)**: Segment documents into semantic chunks strictly $\le 2000$ characters.
2. **Indexing Engine (`src/indexer.py`, `src/embeddings.py`, `src/incremental.py`)**: Persists lexical and vector indices and tracks file modifications.
3. **Retrieval Engine (`src/hybrid.py`, `src/caching.py`)**: Ranks candidate passages using BM25, semantic embeddings, or RRF hybrid scoring with LRU cache acceleration.
4. **Generation Engine (`src/generator.py`)**: Augments questions with extracted context and generates concise answers.
5. **Interfaces (`src/cli.py`, `src/api.py`)**: Exposes functionality through CLI and REST endpoints.

---

## Chunking Strategy

Standard fixed-size sliding windows split methods across arbitrary tokens and lose critical structural boundaries. This project implements two specialized chunking strategies:

### 1. Python Code Chunking
- **AST Parsing**: Analyzes the abstract syntax tree using Python's `ast` module.
- **Module Preamble**: Captures top-level docstrings, imports, and module-level constants that define global settings.
- **Classes**: If a class fits within `max_chunk_size` (2000 chars), it is kept intact. If larger, the class header and class docstring are chunked, followed by individual methods and properties.
- **Functions**: Top-level and class methods are kept intact when $\le 2000$ characters; if oversized, they are sliced with a 15% overlap.
- **Fallback**: If a file has syntax errors, a sliding line window is used gracefully.

### 2. Markdown Chunking
- **Hierarchical Header Parsing**: Decomposes documents into a tree of sections across Markdown levels (`#` through `######`).
- **Multi-Scale Section Indexing**: Emits both focused leaf sections and aggregated parent sections (up to `max_chunk_size`), ensuring that both granular details and broader context are represented.
- **Header Breadcrumb Enrichment**: Injects the active hierarchy path (e.g. `Installation > GPU > CUDA Requirements`) into chunk tokens, bridging the semantic gap when questions reference parent topics.
- **Admonition & Semantic Block Preservation**: Keeps callouts, note blocks (`!!! note`, `??? tip`), and code fences intact without chopping.
- **Natural Paragraph Sliding Windows**: For oversized sections, slices along semantic paragraph boundaries (`\n\s*\n`) with line-level fallback for giant markdown tables, strictly enforcing $\le 2000$ characters.
- **Text & Config Ingestion**: Automatically ingests `.txt` and build scripts like `CMakeLists.txt` referenced in system documentation.

---

## Retrieval Method

### 1. Lexical BM25 Retrieval
- **Code-Aware Tokenization & Morphology**:
  - Splits snake_case (`fused_batched_moe` $\rightarrow$ `fused`, `batched`, `moe`).
  - Splits CamelCase (`ModelRunner` $\rightarrow$ `model`, `runner`).
  - Inflection and suffix normalization (rule-based stemming for plurals, `-ing`, `-ed`, `-tion`) to bridge queries like "vectors" to text "vector".
  - Stopword filtering to reduce noise on common English words while preserving programming keywords.
- **Path & Hierarchy Breadcrumb Boosting**:
  - Incorporates file names, directory components, and Markdown section titles into each chunk's indexed tokens.
  - Queries mentioning module names or section headings receive boosted scores for relevant source files.
- **Type-Aware Scoring**:
  - Contextual doc-type weighting boosts relevant documentation passages on conceptual queries.
- **Okapi BM25 Scoring**:
  - Uses $k_1 = 1.2$ and $b = 0.75$ with document length normalization.

### 2. Semantic Dense Retrieval (Bonus 1)
- Uses `all-MiniLM-L6-v2` to map queries and passages into 384-dimensional normalized dense vectors.
- Calculates cosine similarity via vectorized dot products.

### 3. Reciprocal Rank Fusion (RRF) Hybrid Retrieval (Bonus 2)
Combines sparse BM25 and dense vector rankings:
$$RRF(d) = \frac{w_{lex}}{60 + rank_{lex}(d)} + \frac{w_{sem}}{60 + rank_{sem}(d)}$$
This preserves keyword matching for exact symbols while leveraging semantic similarity for conceptual questions.

---

## Performance Analysis

All benchmarks were evaluated on a CPU-only environment (8 cores, Linux Mint 22.3):

| Metric | Subject Requirement | Achieved System Result | Status |
|---|---|---|---|
| **Docs Recall@5** | $\ge 80.0\%$ | **93.0%** (Recall@10: 95.0%) | **PASS** |
| **Code Recall@5** | $\ge 50.0\%$ | **79.8%** (Recall@10: 88.9%) | **PASS** |
| **Indexing Time** | $\le 5\text{ minutes}$ | **~15.0 seconds** (25,821 chunks) | **PASS** |
| **Retrieval Throughput** | $\le 90\text{ seconds / 200 questions}$ | **~8.0 seconds / 200 questions** | **PASS** |
| **Query Cache Speedup** | N/A (Bonus 4) | **279.6x speedup** (23.5ms $\rightarrow$ 0.084ms) | **PASS** |

### Recall Breakdown
- **Docs Dataset (`dataset_docs_public.json`)**:
  - Recall@1: 73.0%
  - Recall@3: 85.0%
  - Recall@5: 93.0%
  - Recall@10: 95.0%
- **Code Dataset (`dataset_code_public.json`)**:
  - Recall@1: 53.5%
  - Recall@3: 75.8%
  - Recall@5: 79.8%
  - Recall@10: 88.9%

---

## Design Decisions

1. **Sublinear File Path Weighting**: Rather than relying only on passage body text, prepending and weighting path tokens ensures queries that name a file or component (e.g. `triton_flash_attention`, `fused_batched_moe`) prioritize the defining module.
2. **CPU Optimization for Qwen3-0.6B**: On a CPU machine without GPU acceleration, inference speed depends heavily on thread count and sequence length. PyTorch threads were restricted to `min(4, os.cpu_count())` to eliminate OpenMP/MKL lock contention, and prompt context was budgeted to 1800 characters, reducing generation time per answer from minutes to seconds.
3. **Deterministic Multi-Level Caching**: Index loading uses in-memory singletons, and queries use SHA256 parameter hashing to provide sub-millisecond responses on repeated lookups.

---

## Challenges Faced

1. **Module-Level Variable Retrieval**: Initial AST implementations only visited `FunctionDef` and `ClassDef`, causing module-level variables (such as constants `FP8_MIN` and `FP8_MAX`) to be omitted from the index. The AST parser was extended to capture module preambles and non-function expressions.
2. **Path Verification Strictness**: The evaluation moulinette enforces exact character span overlap (IoU $\ge 0.05$) and exact verbatim file paths. Retaining relative path structures (`data/raw/vllm-0.10.1/...`) was critical for valid grading.
3. **Over-Generation in Reasoning Models**: The `Qwen3-0.6B` model can generate internal reasoning tokens if unconstrained. Prompt conditioning and stopping criteria were tuned to produce concise 1-2 sentence answers directly grounded in the context.

---

## Bonus Features

All five bonus features are implemented and fully functional:

1. **Semantic Embeddings (`src/embeddings.py`)**: Dense vector indexing with `all-MiniLM-L6-v2`. Run via `python -m src index --with_embeddings`.
2. **Hybrid Retrieval (`src/hybrid.py`)**: Reciprocal Rank Fusion combining lexical BM25 and dense embeddings. Run via `python -m src search <query> --hybrid`.
3. **Incremental Indexing (`src/incremental.py`)**: Tracks file mtimes and SHA256 hashes. Detects added, modified, and deleted files and re-indexes only changed files without rebuilding the entire corpus. Run via `python -m src index --incremental`.
4. **Caching (`src/caching.py`)**: In-memory singleton index loading and LRU query cache with persistent disk storage (`data/processed/query_cache.json`). Delivers 279x speedup on repeated searches.
5. **Local HTTP API (`src/api.py`)**: FastAPI application exposing `/health`, `/search`, and `/answer` endpoints. Run via `python -m src serve --port 8000`.

---

## Instructions

### 1. Installation

Install dependencies using `uv`:

```bash
make install
# or
uv sync
```

### 2. Building the Index

Index the corpus (default: `data/raw`, `max_chunk_size=2000`):

```bash
uv run python -m src index --max_chunk_size 2000
```

To build both lexical and semantic vector indices:

```bash
uv run python -m src index --max_chunk_size 2000 --with_embeddings
```

To update only changed files incrementally:

```bash
uv run python -m src index --incremental
```

### 3. Searching

Single query search:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

Hybrid search (lexical + semantic embeddings):

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5 --hybrid
```

Batch dataset search:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets_public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

### 4. Answer Generation

Single query answer:

```bash
uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?" --k 3
```

Batch dataset answer generation:

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### 5. Evaluation

Evaluate against ground truth with the local evaluate command:

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets_public/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

Evaluate with the official moulinette:

```bash
./moulinette/moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets_public/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```

### 6. Local HTTP API Server

Start the REST API server:

```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

Query the API via `curl`:

```bash
# Health check
curl http://127.0.0.1:8000/health

# Search endpoint
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "OpenAI server", "k": 3}'

# Answer endpoint
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "How to configure OpenAI server?", "k": 3}'
```

### 7. Code Quality and Testing

Execute linter checks:

```bash
make lint
```

Strict type checking:

```bash
make lint-strict
```

Clean temporary files and outputs:

```bash
make clean
```

---

## Resources

- **BM25 & Information Retrieval**:
  - Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
  - Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR '09.
- **Language Models & Sentence Transformers**:
  - Qwen Team. *Qwen Technical Report*. Alibaba Cloud.
  - Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
- **vLLM Project**:
  - Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.

### AI Usage

AI assistance was utilized in accordance with Chapter III (AI Instructions):
- **Boilerplate and Structure**: Drafting initial Pydantic schema validation structures and FastAPI route scaffolding.
- **Algorithm Exploration**: Reviewing parameter ranges for Okapi BM25 and Reciprocal Rank Fusion formulas.
- **Code Review**: Identifying missed AST nodes (such as module constants in `chunking.py`) and debugging static typing edge cases with `mypy`.
- **Validation**: All AI-assisted components were manually reviewed, rewritten where needed, benchmarked on CPU, and tested against the project's evaluation moulinette.
