from models import *
from index import read_file, chunk_md


MAX_CHUNK_SIZE = 2000

if __name__ == "__main__":
    filepath = "./data/raw/vllm-0.10.1/README.md"
    line = read_file(filepath)
    minimal_sources = chunk_md(line, filepath)
    print(minimal_sources[0])