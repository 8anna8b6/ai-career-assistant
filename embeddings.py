from __future__ import annotations
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import LOCAL_EMBEDDING_MODEL

_model = None  # loaded once


def _init_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"[Embeddings] Using local model: {LOCAL_EMBEDDING_MODEL}")


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

    vectors = _model.encode(
        cleaned,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True  
    )

    return [v.tolist() for v in vectors]


def embedding_dim() -> int:
    _init_model()
    return len(get_embeddings(["test"])[0])