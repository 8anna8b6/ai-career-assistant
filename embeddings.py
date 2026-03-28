from __future__ import annotations
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config import LOCAL_EMBEDDING_MODEL

_model = None  # The AI model (loaded once)

def _init_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        print(f"[Embeddings] Using local model: {LOCAL_EMBEDDING_MODEL}")



def get_embeddings(texts: Union[str, List[str]]) -> List[List[float]]:
    _init_model()

    if not texts:
        return []

    # Ensure list input
    if isinstance(texts, str):
        texts = [texts]

    # Flatten nested lists 
    flattened = []
    for t in texts:
        if isinstance(t, list):
            flattened.extend(t)
        else:
            flattened.append(t)

    # Clean + normalize
    cleaned_texts = []
    for t in flattened:
        if not isinstance(t, str):
            t = str(t)

        t = t.strip()
        cleaned_texts.append(t if t else " ")

    # Generate embeddings
    vectors = _model.encode(
        cleaned_texts,
        show_progress_bar=False,
        batch_size=32
    )

    return [v.tolist() for v in vectors]


def embedding_dim() -> int:
    _init_model()
    return len(get_embeddings(["test"])[0])