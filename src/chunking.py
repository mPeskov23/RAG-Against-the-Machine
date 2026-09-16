from .models import MinimalSource
import ast


class CodeChunker(ast.NodeVisitor):
    def __init__(self,
                 file_path: str,
                 line_offsets: list[int],
                 max_chunk_size: int = 2000) -> None:
        self.file_path = file_path
        self.line_offsets = line_offsets
        self.max_chunk_size = max_chunk_size
        self.chunks: list[MinimalSource] = []

    def _process_node(self, node: ast.AST) -> None:
        first_idx, last_idx = get_char_indices(node, self.line_offsets)
        chunk_len = last_idx - first_idx
        if chunk_len <= self.max_chunk_size and chunk_len > 0:
            self.chunks.append(
                MinimalSource(
                    file_path=self.file_path,
                    first_character_index=first_idx,
                    last_character_index=last_idx
                )
            )
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_node(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._process_node(node)


def read_file(filename: str) -> str:
    ret_str = ""
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            ret_str = f.read()
        except Exception:
            print(f"Error reading {filename}")
    return ret_str


def chunk_md(file_path: str, max_chunk_size: int = 2000) -> list[MinimalSource]:
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
        source.file_path = file_path
        source.first_character_index = start
        source.last_character_index = end
        ret_list.append(source)
        start += max_chunk_size - overlap
    return ret_list


def get_line_offsets(text: str) -> list[int]:
    offsets: list[int] = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def get_char_indices(node: ast.AST, line_offsets: list[int]) -> tuple[int, int]:
    start_line = getattr(node, "lineno", 1) - 1
    start_col = getattr(node, "col_offset", 0)
    first_idx = line_offsets[start_line] + start_col
    end_line = getattr(node, "end_lineno", getattr(node, "lineno", 1)) - 1
    end_col = getattr(node, "end_col_offset", 0)
    last_idx = line_offsets[end_line] + end_col

    return first_idx, last_idx


def chunk_py(filename: str, max_chunk_size: int = 2000) -> list[MinimalSource]:
    text = read_file(filename)
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        print(f"Syntax error in {filename}, falling back to text chunking")
        return chunk_md(text, filename, max_chunk_size)
    line_offsets = get_line_offsets(text)
    chunker = CodeChunker(file_path=filename, line_offsets=line_offsets, max_chunk_size=max_chunk_size)
    chunker.visit(tree)
    if not chunker.chunks:
        return chunk_md(text, filename, max_chunk_size)
    return chunker.chunks


def chunk(filename: str, max_chunk_size: int = 2000) -> list[MinimalSource]:
    if filename.endswith('.py'):
        return chunk_py(filename, max_chunk_size)
    else:
        return chunk_md(filename, max_chunk_size)
