import ast
from .models import MinimalSource


class CodeChunker(ast.NodeVisitor):
    def __init__(self,
                 file_path: str,
                 line_offsets: list[int],
                 max_chunk_size: int = 2000) -> None:
        self.file_path = file_path
        self.line_offsets = line_offsets
        self.max_chunk_size = max_chunk_size
        self.chunks: list[MinimalSource] = []

    def _add_chunk(self, first_idx: int, last_idx: int) -> None:
        if last_idx > first_idx:
            self.chunks.append(
                MinimalSource(
                    file_path=self.file_path,
                    first_character_index=first_idx,
                    last_character_index=last_idx
                )
            )

    def _slice_range(self, first_idx: int, last_idx: int) -> None:
        overlap = int(self.max_chunk_size / 10)
        step = max(1, self.max_chunk_size - overlap)
        start = first_idx
        while start < last_idx:
            end = min(start + self.max_chunk_size, last_idx)
            self._add_chunk(start, end)
            if end >= last_idx:
                break
            start += step

    def _process_func(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        first_idx, last_idx = get_char_indices(node, self.line_offsets)
        chunk_len = last_idx - first_idx
        if 0 < chunk_len <= self.max_chunk_size:
            self._add_chunk(first_idx, last_idx)
        elif chunk_len > self.max_chunk_size:
            self._slice_range(first_idx, last_idx)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_func(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        first_idx, last_idx = get_char_indices(node, self.line_offsets)
        chunk_len = last_idx - first_idx
        if 0 < chunk_len <= self.max_chunk_size:
            self._add_chunk(first_idx, last_idx)
        else:
            prev_chunks = len(self.chunks)
            self.generic_visit(node)
            if len(self.chunks) == prev_chunks and chunk_len > 0:
                self._slice_range(first_idx, last_idx)

    def visit_If(self, node: ast.If) -> None:
        first_idx, last_idx = get_char_indices(node, self.line_offsets)
        chunk_len = last_idx - first_idx
        if 50 <= chunk_len <= self.max_chunk_size:
            self._add_chunk(first_idx, last_idx)
        elif chunk_len > self.max_chunk_size:
            self._slice_range(first_idx, last_idx)


def read_file(filename: str) -> str:
    ret_str = ""
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            ret_str = f.read()
        except Exception:
            print(f"Error reading {filename}")
    return ret_str


def chunk_md(
    file_path: str, max_chunk_size: int = 2000
) -> list[MinimalSource]:
    text = read_file(file_path)
    ret_list: list[MinimalSource] = []
    overlap = int(max_chunk_size / 10)
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        if end > len(text):
            end = len(text)
        source: MinimalSource = MinimalSource(file_path=file_path,
                                              first_character_index=start,
                                              last_character_index=end)
        ret_list.append(source)
        start += max_chunk_size - overlap
    return ret_list


def get_line_offsets(text: str) -> list[int]:
    offsets: list[int] = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def get_char_indices(
    node: ast.AST, line_offsets: list[int]
) -> tuple[int, int]:
    start_line = getattr(node, "lineno", 1) - 1
    start_col = getattr(node, "col_offset", 0)
    first_idx = line_offsets[start_line] + start_col
    end_line = getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1
    end_col = getattr(node, "end_col_offset", 0)
    last_idx = line_offsets[end_line] + end_col

    return first_idx, last_idx


def chunk_py(
    filename: str, max_chunk_size: int = 2000
) -> list[MinimalSource]:
    text = read_file(filename)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        print(f"Syntax error in {filename}, falling back to text chunking")
        return chunk_md(filename, max_chunk_size)
    line_offsets = get_line_offsets(text)
    chunker = CodeChunker(
        file_path=filename,
        line_offsets=line_offsets,
        max_chunk_size=max_chunk_size
    )
    chunker.visit(tree)
    if not chunker.chunks:
        return chunk_md(filename, max_chunk_size)
    return chunker.chunks


def chunk(filename: str, max_chunk_size: int = 2000) -> list[MinimalSource]:
    if filename.endswith('.py'):
        return chunk_py(filename, max_chunk_size)
    else:
        return chunk_md(filename, max_chunk_size)
