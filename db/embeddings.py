from __future__ import annotations
import logging
from typing import List, Union
from chromadb.utils import embedding_functions

log = logging.getLogger(__name__)

_model = None

def _init_model():
    global _model
    if _model is None:
        _model = embedding_functions.DefaultEmbeddingFunction()
        log.info("Embedding model loaded (ChromaDB default).")


def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    _init_model()

    if not texts:
        return []

    if isinstance(texts, str):
        texts = [texts]

    # Flatten nested lists and clean empty strings
    cleaned = []
    for t in texts:
        if isinstance(t, list):
            cleaned.extend(str(v).strip() or " " for v in t)
        else:
            cleaned.append(str(t).strip() or " ")

    return _model(cleaned)


def embedding_dim() -> int:
    return len(get_embeddings(["test"])[0])