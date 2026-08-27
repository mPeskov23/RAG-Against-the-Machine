from models import MinimalSource


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
