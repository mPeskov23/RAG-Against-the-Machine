from models import MinimalSource
import ast


def read_file(filename: str) -> str:
    ret_str = ""
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            ret_str = f.read()
        except Exception:
            print(f"Error reading {filename}")
    return ret_str


def chunk_md(text: str, file_path: str, max_chunk_size: int = 2000) -> list[MinimalSource]:
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


def parse_py(filename: str) -> ast.Module | None:
    text = read_file(filename)
    line_offsets = get_line_offsets(text)
    try:
        tree = ast.parse(text)
        return tree
    except SyntaxError:
        print(f"Some syntax error in python file {filename}")
    return None


def chunk_py(filename: str) -> list[MinimalSource]:
    retlist: list[MinimalSource] = []
    tree: ast.Module | None = parse_py(filename)
    if tree is None:
        return retlist
    line_offsets: list[int] = get_line_offsets(filname)