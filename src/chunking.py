"""Document and code chunking module.

Implements two distinct chunking strategies:
- Python AST-based semantic chunking (functions, classes, methods, module constants)
- Markdown hierarchical header and paragraph-based chunking
"""

import ast
import re
from typing import List, Tuple
from .models import MinimalSource


def read_file(filename: str) -> str:
    """Read full text content of a file using utf-8 with fallback error replacement."""
    try:
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def get_line_offsets(text: str) -> List[int]:
    """Calculate character offset for the beginning of each line."""
    offsets: List[int] = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def get_node_span(node: ast.AST, line_offsets: List[int], total_len: int) -> Tuple[int, int]:
    """Calculate exact start and end character offsets for an AST node."""
    start_line = max(0, getattr(node, "lineno", 1) - 1)
    start_col = getattr(node, "col_offset", 0)
    first_idx = min(line_offsets[start_line] + start_col, total_len)

    end_line = max(start_line, getattr(node, "end_lineno", start_line + 1) - 1)
    if end_line < len(line_offsets):
        end_col = getattr(node, "end_col_offset", 0)
        last_idx = min(line_offsets[end_line] + end_col, total_len)
    else:
        last_idx = total_len

    return first_idx, max(first_idx, last_idx)


def _slice_window(
    first_idx: int, last_idx: int, max_chunk_size: int, overlap_ratio: float = 0.15
) -> List[Tuple[int, int]]:
    """Slice a large range into overlapping sub-ranges strictly <= max_chunk_size."""
    if last_idx <= first_idx:
        return []
    if last_idx - first_idx <= max_chunk_size:
        return [(first_idx, last_idx)]

    slices: List[Tuple[int, int]] = []
    overlap = max(50, int(max_chunk_size * overlap_ratio))
    step = max(1, max_chunk_size - overlap)
    start = first_idx

    while start < last_idx:
        end = min(start + max_chunk_size, last_idx)
        slices.append((start, end))
        if end >= last_idx:
            break
        start += step

    return slices


def chunk_py(file_path: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """Chunk Python files using AST node analysis and statement boundaries."""
    text = read_file(file_path)
    total_len = len(text)
    if not text.strip():
        return []

    try:
        tree = ast.parse(text, filename=file_path)
    except SyntaxError:
        return chunk_md(file_path, max_chunk_size=max_chunk_size)

    line_offsets = get_line_offsets(text)
    spans: List[Tuple[int, int]] = []

    if not tree.body:
        spans.extend(_slice_window(0, total_len, max_chunk_size))
    else:
        # Group non-class/non-function statements at module level (constants, docstrings, imports)
        pending_start: int | None = None
        pending_end: int | None = None

        for stmt in tree.body:
            s_idx, e_idx = get_node_span(stmt, line_offsets, total_len)

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Flush pending module-level statements
                if pending_start is not None and pending_end is not None:
                    spans.extend(_slice_window(pending_start, pending_end, max_chunk_size))
                    pending_start, pending_end = None, None

                fn_len = e_idx - s_idx
                if fn_len <= max_chunk_size:
                    spans.append((s_idx, e_idx))
                else:
                    spans.extend(_slice_window(s_idx, e_idx, max_chunk_size))

            elif isinstance(stmt, ast.ClassDef):
                # Flush pending module-level statements
                if pending_start is not None and pending_end is not None:
                    spans.extend(_slice_window(pending_start, pending_end, max_chunk_size))
                    pending_start, pending_end = None, None

                cls_len = e_idx - s_idx
                if cls_len <= max_chunk_size:
                    spans.append((s_idx, e_idx))
                else:
                    # Capture class header + docstring + class attributes
                    cls_header_end = min(e_idx, s_idx + min(max_chunk_size, 1000))
                    spans.append((s_idx, cls_header_end))

                    # Chunk class methods and nested definitions
                    for item in stmt.body:
                        item_s, item_e = get_node_span(item, line_offsets, total_len)
                        item_len = item_e - item_s
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item_len <= max_chunk_size:
                                spans.append((item_s, item_e))
                            else:
                                spans.extend(_slice_window(item_s, item_e, max_chunk_size))
                        elif item_len > 100:
                            spans.extend(_slice_window(item_s, item_e, max_chunk_size))

            else:
                # Accumulate module-level assignments, imports, if statements, docstrings
                if pending_start is None:
                    pending_start = s_idx
                pending_end = e_idx

        if pending_start is not None and pending_end is not None:
            spans.extend(_slice_window(pending_start, pending_end, max_chunk_size))

    # Convert spans to MinimalSource, deduplicate and ensure non-empty
    seen = set()
    chunks: List[MinimalSource] = []
    for s_idx, e_idx in spans:
        s_idx = max(0, min(s_idx, total_len))
        e_idx = max(s_idx, min(e_idx, total_len))
        if e_idx - s_idx > 0 and (s_idx, e_idx) not in seen:
            seen.add((s_idx, e_idx))
            # Ensure chunk is strictly within max_chunk_size
            if e_idx - s_idx > max_chunk_size:
                e_idx = s_idx + max_chunk_size
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=s_idx,
                    last_character_index=e_idx,
                )
            )

    if not chunks:
        # Fallback to sliding window
        for s, e in _slice_window(0, total_len, max_chunk_size):
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                )
            )

    return chunks


def chunk_md(file_path: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """Chunk Markdown files using header hierarchy and paragraph boundaries."""
    text = read_file(file_path)
    total_len = len(text)
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    line_offsets = get_line_offsets(text)
    header_indices: List[int] = []

    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+", line):
            header_indices.append(i)

    spans: List[Tuple[int, int]] = []

    if not header_indices:
        # No headers found; split by paragraph boundaries
        paragraphs = [m.start() for m in re.finditer(r"\n\s*\n", text)]
        if not paragraphs:
            spans.extend(_slice_window(0, total_len, max_chunk_size))
        else:
            p_starts = [0] + [p + 1 for p in paragraphs]
            p_starts.append(total_len)
            curr_start = 0
            for p_end in p_starts[1:]:
                if p_end - curr_start > max_chunk_size:
                    if curr_start < p_end:
                        spans.extend(_slice_window(curr_start, p_end, max_chunk_size))
                    curr_start = p_end
                else:
                    pass
            if curr_start < total_len:
                spans.extend(_slice_window(curr_start, total_len, max_chunk_size))
    else:
        # Slice into header sections
        sections: List[Tuple[int, int]] = []
        if header_indices[0] > 0:
            sections.append((0, line_offsets[header_indices[0]]))

        for idx, h_idx in enumerate(header_indices):
            s_char = line_offsets[h_idx]
            if idx + 1 < len(header_indices):
                e_char = line_offsets[header_indices[idx + 1]]
            else:
                e_char = total_len
            sections.append((s_char, e_char))

        for s_char, e_char in sections:
            sec_len = e_char - s_char
            if sec_len <= 0:
                continue
            if sec_len <= max_chunk_size:
                spans.append((s_char, e_char))
            else:
                spans.extend(_slice_window(s_char, e_char, max_chunk_size))

    seen = set()
    chunks: List[MinimalSource] = []
    for s_idx, e_idx in spans:
        s_idx = max(0, min(s_idx, total_len))
        e_idx = max(s_idx, min(e_idx, total_len))
        if e_idx - s_idx > 0 and (s_idx, e_idx) not in seen:
            seen.add((s_idx, e_idx))
            if e_idx - s_idx > max_chunk_size:
                e_idx = s_idx + max_chunk_size
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=s_idx,
                    last_character_index=e_idx,
                )
            )

    return chunks


def chunk(filename: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """Route file to appropriate chunker based on file extension."""
    if filename.endswith(".py"):
        return chunk_py(filename, max_chunk_size=max_chunk_size)
    return chunk_md(filename, max_chunk_size=max_chunk_size)
