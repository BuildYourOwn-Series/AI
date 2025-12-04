# chap08/knowledge_store.json

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import re
import numpy as np

from embedding_backend import EmbeddingBackend

FLAG_RE = re.compile(r"--?[A-Za-z][A-Za-z0-9]*")

def extract_flags(cmd: str) -> List[str]:
    return FLAG_RE.findall(cmd)

@dataclass
class KnowledgeEntry:
    key: str
    description: str
    command: str
    vector: np.ndarray


class KnowledgeStore:
    def __init__(self, backend: EmbeddingBackend,
                 entries: List[Tuple[str, str, str]]) -> None:
        """
        entries: list of (key, description, command) triples.
        """
        self.backend = backend
        self.entries: List[KnowledgeEntry] = []
        for key, desc, cmd in entries:
            text = desc + " :: " + cmd
            vec = backend.encode(text)
            self.entries.append(
                KnowledgeEntry(key=key, description=desc,
                               command=cmd, vector=vec)
            )

    def query(self, text: str, k: int = 3) -> List[KnowledgeEntry]:
        """
        Return the top-k entries most similar to the query text.
        """
        q_vec = self.backend.encode(text)
        if not np.any(q_vec):
            return []

        scores: List[Tuple[float, KnowledgeEntry]] = []
        for e in self.entries:
            # Cosine similarity becomes a dot product thanks to L2 norm.
            score = float(np.dot(q_vec, e.vector))
            scores.append((score, e))

        scores.sort(key=lambda t: t[0], reverse=True)
        return [e for score, e in scores[:k] if score > 0.0]


def build_default_store() -> Tuple[EmbeddingBackend, KnowledgeStore]:
    """
    Convenience helper: build a tiny default corpus and store.
    """
    raw_entries: List[Tuple[str, str, str]] = [
        ("ls_docs",
         "ls lists directory contents. Common flags include -l (details) "
         "and -a (show hidden files).",
         "ls -la"),
        ("grep_docs",
         "grep searches for text patterns in files. -R searches recursively.",
         "grep -R pattern"),
        ("find_docs",
         "find walks the filesystem and matches files by name, size, or type.",
         "find . -type f"),
        ("list_files",
         "List files in the current directory with details",
         "ls -la"),
        ("find_large_files",
         "Find regular files larger than 100 megabytes",
         "find . -type f -size +100M"),
        ("search_text",
         "Recursively search for a string in source files",
         "grep -R \"pattern\" src/"),
        ("disk_usage",
         "Show human-readable disk usage for subdirectories",
         "du -sh * | sort -h"),
        ("remove_tree",
         "Recursively remove a directory tree (dangerous!)",
         "rm -rf path/to/dir"),
    ]

    # Build backend from all descriptions + commands.
    corpus = [d + " " + c for _, d, c in raw_entries]
    backend = EmbeddingBackend.from_corpus(corpus)
    store = KnowledgeStore(backend, raw_entries)
    return backend, store

