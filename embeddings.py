from __future__ import annotations
from typing import List, Union
import chromadb
from chromadb.utils import embedding_functions

_model = None  # loaded once


def _init_model():
    global _model
    if _model is None:
        # Use ChromaDB's default embedding (lightweight, no torch needed)
        _model = embedding_functions.DefaultEmbeddingFunction()
        print(f"[Embeddings] Using ChromaDB default embedding function")


def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    _init_model()

    if not texts:
        return []

    if isinstance(texts, str):
        texts = [texts]

    # flatten
    flattened = []
    for t in texts:
        if isinstance(t, list):
            flattened.extend(t)
        else:
            flattened.append(t)

    # clean
    cleaned = []
    for t in flattened:
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        cleaned.append(t if t else " ")

    # Get embeddings using ChromaDB's default function
    vectors = _model(cleaned)
    
    return vectors


def embedding_dim() -> int:
    _init_model()
    return len(get_embeddings(["test"])[0])