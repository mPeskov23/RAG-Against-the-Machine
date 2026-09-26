"""Document and code chunking module.

Implements two distinct chunking strategies:
- Python AST-based semantic chunking
  (functions, classes, methods, module constants)
- Markdown hierarchical header and paragraph-based chunking
"""

import ast
import re
from typing import List, Tuple
from .models import MinimalSource


def read_file(filename: str) -> str:
    """Read full text content of a file using utf-8
    with fallback error replacement."""
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


def get_node_span(
    node: ast.AST, line_offsets: List[int], total_len: int
) -> Tuple[int, int]:
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
    first_idx: int,
    last_idx: int,
    max_chunk_size: int,
    overlap_ratio: float = 0.15,
) -> List[Tuple[int, int]]:
    """Slice a large range into overlapping sub-ranges
    strictly <= max_chunk_size."""
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


def chunk_py(
    file_path: str, max_chunk_size: int = 2000
) -> List[MinimalSource]:
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
        # Group non-class/non-function statements at module level
        # (constants, docstrings, imports)
        pending_start: int | None = None
        pending_end: int | None = None

        for stmt in tree.body:
            s_idx, e_idx = get_node_span(stmt, line_offsets, total_len)

            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Flush pending module-level statements
                if pending_start is not None and pending_end is not None:
                    spans.extend(
                        _slice_window(
                            pending_start, pending_end, max_chunk_size
                        )
                    )
                    pending_start, pending_end = None, None

                fn_len = e_idx - s_idx
                if fn_len <= max_chunk_size:
                    spans.append((s_idx, e_idx))
                else:
                    spans.extend(_slice_window(s_idx, e_idx, max_chunk_size))

            elif isinstance(stmt, ast.ClassDef):
                # Flush pending module-level statements
                if pending_start is not None and pending_end is not None:
                    spans.extend(
                        _slice_window(
                            pending_start, pending_end, max_chunk_size
                        )
                    )
                    pending_start, pending_end = None, None

                cls_len = e_idx - s_idx
                if cls_len <= max_chunk_size:
                    spans.append((s_idx, e_idx))
                else:
                    # Capture class header + docstring + class attributes
                    cls_header_end = min(
                        e_idx, s_idx + min(max_chunk_size, 1000)
                    )
                    spans.append((s_idx, cls_header_end))

                    # Chunk class methods and nested definitions
                    for item in stmt.body:
                        item_s, item_e = get_node_span(
                            item, line_offsets, total_len
                        )
                        item_len = item_e - item_s
                        if isinstance(
                            item, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            if item_len <= max_chunk_size:
                                spans.append((item_s, item_e))
                            else:
                                spans.extend(
                                    _slice_window(
                                        item_s, item_e, max_chunk_size
                                    )
                                )
                        elif item_len > 100:
                            spans.extend(
                                _slice_window(item_s, item_e, max_chunk_size)
                            )

            else:
                # Accumulate module-level assignments, imports,
                # if statements, docstrings
                if pending_start is None:
                    pending_start = s_idx
                pending_end = e_idx

        if pending_start is not None and pending_end is not None:
            spans.extend(
                _slice_window(pending_start, pending_end, max_chunk_size)
            )

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


def _split_blocks_with_overlap(
    text: str,
    start_char: int,
    end_char: int,
    max_chunk_size: int = 2000,
    target_overlap: int = 300,
) -> List[Tuple[int, int]]:
    """Split a section of text into overlapping block chunks cleanly."""
    sec_text = text[start_char:end_char]
    p_matches = list(re.finditer(r"\n\s*\n", sec_text))

    if not p_matches:
        spans: List[Tuple[int, int]] = []
        c_start = 0
        while c_start < len(sec_text):
            c_end = min(len(sec_text), c_start + max_chunk_size)
            if c_end < len(sec_text):
                last_nl = sec_text.rfind("\n", c_start, c_end)
                if last_nl > c_start:
                    c_end = last_nl + 1
            spans.append((start_char + c_start, start_char + c_end))
            if c_end >= len(sec_text):
                break
            c_start += max(200, max_chunk_size - target_overlap)
        return spans

    b_starts = [0] + [m.end() for m in p_matches]
    b_ends = [m.start() for m in p_matches] + [len(sec_text)]
    blocks = [(s, e) for s, e in zip(b_starts, b_ends) if e > s]

    spans = []
    curr_i = 0
    while curr_i < len(blocks):
        s_idx = blocks[curr_i][0]
        end_i = curr_i
        while end_i < len(blocks) and (
            blocks[end_i][1] - s_idx
        ) <= max_chunk_size:
            end_i += 1

        if end_i == curr_i:
            bs, be = blocks[curr_i]
            c_start = bs
            while c_start < be:
                c_end = min(be, c_start + max_chunk_size)
                if c_end < be:
                    last_nl = sec_text.rfind("\n", c_start, c_end)
                    if last_nl > c_start:
                        c_end = last_nl + 1
                spans.append((start_char + c_start, start_char + c_end))
                if c_end >= be:
                    break
                c_start += max(200, max_chunk_size - target_overlap)
            curr_i += 1
        else:
            e_idx = blocks[end_i - 1][1]
            spans.append((start_char + s_idx, start_char + e_idx))
            if end_i >= len(blocks):
                break
            next_i = end_i
            while next_i > curr_i + 1 and (
                e_idx - blocks[next_i - 1][0]
            ) < target_overlap:
                next_i -= 1
            curr_i = next_i

    return spans


def chunk_md(
    file_path: str, max_chunk_size: int = 2000
) -> List[MinimalSource]:
    """Chunk Markdown files using hierarchical headers and semantic blocks."""
    text = read_file(file_path)
    total_len = len(text)
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    line_offsets = get_line_offsets(text)

    headers: List[Tuple[int, int, str]] = []
    raw_spans: List[Tuple[int, int]] = []

    def add_span(s: int, e: int) -> None:
        s = max(0, min(s, total_len))
        e = max(s, min(e, total_len))
        while e > s and text[e - 1] in " \t\r\n":
            e -= 1
        while s < e and text[s] in " \t\r\n":
            s += 1
        if 1 <= (e - s) <= max_chunk_size:
            raw_spans.append((s, e))

    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            s_offset = line_offsets[i]
            e_offset = s_offset + len(line.rstrip())
            headers.append(
                (s_offset, len(m.group(1)), m.group(2).strip())
            )
            add_span(s_offset, e_offset)

    # Emit anchor links (e.g. [](){ #chat-template } or <a id="...">)
    anchor_pattern = (
        r"\[\]\(\)\{\s*#[^}]+\s*\}|<a\s+id=[\'\"][^\'\"]+[\'\"]|\{:\s*#[^}]+\}"
    )
    for m in re.finditer(anchor_pattern, text):
        add_span(m.start(), m.end())

    if not headers:
        for s, e in _split_blocks_with_overlap(
            text, 0, total_len, max_chunk_size=max_chunk_size
        ):
            add_span(s, e)
    else:
        if headers[0][0] > 0:
            add_span(0, headers[0][0])

        for i in range(len(headers)):
            s_char = headers[i][0]
            e_char = headers[i + 1][0] if i + 1 < len(headers) else total_len
            sec_len = e_char - s_char
            if sec_len <= max_chunk_size:
                add_span(s_char, e_char)
            else:
                for s, e in _split_blocks_with_overlap(
                    text, s_char, e_char, max_chunk_size=max_chunk_size
                ):
                    add_span(s, e)

        for i in range(len(headers)):
            s_char, lvl, _ = headers[i]
            e_char = total_len
            for j in range(i + 1, len(headers)):
                if headers[j][1] <= lvl:
                    e_char = headers[j][0]
                    break
            if e_char - s_char <= max_chunk_size:
                add_span(s_char, e_char)

        for m in re.finditer(
            r"^\s*(?:!{3}|\?{3})\s+.*?(?=\n\n|\Z)",
            text,
            flags=re.DOTALL | re.MULTILINE,
        ):
            add_span(m.start(), m.end())

    seen = set()
    chunks: List[MinimalSource] = []
    for s, e in sorted(raw_spans):
        if (s, e) not in seen:
            seen.add((s, e))
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                )
            )

    if not chunks:
        for s, e in _slice_window(0, total_len, max_chunk_size):
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                )
            )

    return chunks


def chunk(filename: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """Route file to appropriate chunker based on file extension."""
    if filename.endswith(".py"):
        return chunk_py(filename, max_chunk_size=max_chunk_size)
    return chunk_md(filename, max_chunk_size=max_chunk_size)
