"""Incremental indexing module.

Bonus 3: Detect modified, added, and deleted files and re-index only changed files
instead of rebuilding the whole index.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
from .chunking import chunk
from .indexer import BM25Indexer
from .models import MinimalSource


METADATA_FILE = "data/processed/incremental_metadata.json"


def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk_bytes := f.read(65536):
                hasher.update(chunk_bytes)
        return hasher.hexdigest()
    except Exception:
        return ""


class IncrementalIndexManager:
    """Manages file state tracking and partial re-indexing of modified files."""

    def __init__(self, metadata_path: str | Path = METADATA_FILE) -> None:
        """Initialize incremental index manager with metadata state path."""
        self.metadata_path = Path(metadata_path)
        self.registry: Dict[str, Dict[str, float | str | int]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, float | str | int]]:
        """Load file state registry from disk or return empty dict."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return cast(Dict[str, Dict[str, float | str | int]], data)
            except Exception:
                pass
        return {}

    def save_registry(self) -> None:
        """Persist file state registry to disk."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def scan_changes(
        self, corpus_dir: str | Path
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """Scan corpus directory and detect added, modified, and deleted files."""
        base = Path(corpus_dir)
        current_files: Dict[str, Path] = {
            str(p): p
            for p in base.rglob("*")
            if p.is_file() and p.suffix in [".py", ".md"]
        }

        added: List[Path] = []
        modified: List[Path] = []
        deleted: List[str] = []

        registered_paths: Set[str] = set(self.registry.keys())

        # Check for deleted files
        for reg_path in registered_paths:
            if reg_path not in current_files:
                deleted.append(reg_path)

        # Check for added or modified files
        for path_str, p in current_files.items():
            if path_str not in self.registry:
                added.append(p)
            else:
                curr_mtime = p.stat().st_mtime
                saved_mtime = float(self.registry[path_str].get("mtime", 0.0))
                if curr_mtime != saved_mtime:
                    curr_hash = get_file_hash(p)
                    if curr_hash != self.registry[path_str].get("hash", ""):
                        modified.append(p)

        return added, modified, deleted

    def update_index(
        self,
        indexer: BM25Indexer,
        corpus_dir: str | Path,
        max_chunk_size: int = 2000,
    ) -> Tuple[int, int, int]:
        """Incrementally re-index only added, modified, or deleted files."""
        added, modified, deleted = self.scan_changes(corpus_dir)
        changed_paths: Set[str] = {str(p) for p in modified} | set(deleted)

        if not added and not modified and not deleted and self.registry:
            return 0, 0, 0

        # Filter out old chunks for modified and deleted files
        preserved_corpus: List[MinimalSource] = [
            src for src in indexer.corpus if src.file_path not in changed_paths
        ]

        # Chunk only added and modified files
        new_chunks: List[MinimalSource] = []
        files_to_chunk = added + modified

        for p in files_to_chunk:
            file_str = str(p)
            file_chunks = chunk(file_str, max_chunk_size=max_chunk_size)
            new_chunks.extend(file_chunks)
            self.registry[file_str] = {
                "mtime": p.stat().st_mtime,
                "hash": get_file_hash(p),
                "chunks": len(file_chunks),
            }

        # Remove deleted files from registry
        for del_path in deleted:
            self.registry.pop(del_path, None)

        indexer.corpus = preserved_corpus + new_chunks
        indexer.fit()

        self.save_registry()
        return len(added), len(modified), len(deleted)
