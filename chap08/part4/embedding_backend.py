# chap08/ai_shell_minimal.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np


def tokenize(text: str) -> List[str]:
    """Very simple whitespace + punctuation tokenizer."""
    # Lowercase and replace common punctuation with spaces.
    chars = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append(" ")
    return [tok for tok in "".join(chars).split() if tok]


@dataclass
class EmbeddingBackend:
    vocab: Dict[str, int]
    idf: np.ndarray

    @classmethod
    def from_corpus(cls, documents: List[str]) -> "EmbeddingBackend":
        """
        Build vocabulary and IDF weights from a small corpus.
        documents: list of texts (commands, descriptions, docs).
        """
        # Build vocabulary.
        vocab: Dict[str, int] = {}
        doc_tokens: List[List[str]] = []
        for doc in documents:
            tokens = tokenize(doc)
            doc_tokens.append(tokens)
            for tok in set(tokens):
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        # Document frequency for each term.
        df = np.zeros(len(vocab), dtype=np.float32)
        for tokens in doc_tokens:
            seen = set()
            for tok in tokens:
                idx = vocab[tok]
                if idx not in seen:
                    df[idx] += 1.0
                    seen.add(idx)

        n_docs = float(len(documents))
        # Standard idf: log((N + 1) / (df + 1)) + 1
        idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        return cls(vocab=vocab, idf=idf)

    def encode(self, text: str) -> np.ndarray:
        """
        Map text to a d-dimensional L2-normalised tf-idf vector.
        """
        tokens = tokenize(text)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        for tok in tokens:
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0

        if not np.any(vec):
            return vec  # all zeros for completely unknown text

        # Term frequency (raw counts) scaled by idf.
        vec *= self.idf

        # L2 normalisation so that dot products approximate cosine sim.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

