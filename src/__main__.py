from models import *
from index import read_file, chunk_md, read_py
import ast


MAX_CHUNK_SIZE = 2000

if __name__ == "__main__":
    filepath = "./src/__main__.py"
    tree = read_py(filepath)
    print(ast.dump(tree))