from __future__ import annotations
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from embeddings import get_embeddings
from utils import build_embedding_text, build_chroma_metadata

IMPORTANT_FIELDS = [
    "skills_must",
    "skills_nice",
    "tools_technologies",
    "description",
    "requirements",
    "title",
    "role",
    "company",
    "location"
]

def init_chroma() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()
    print(f"[ChromaDB] Collection '{CHROMA_COLLECTION}' | {count} vectors | path: {CHROMA_PERSIST_DIR}")
    return collection


def upsert_jobs(collection: chromadb.Collection, jobs: List[dict]) -> int:
    if not jobs:
        return 0

    total_ids = []
    for job in jobs:
        meta = build_chroma_metadata(job)
        for field in IMPORTANT_FIELDS:
            val = job.get(field, "") or ""
            # Convert lists to comma-separated strings
            if isinstance(val, list):
                doc_text = ", ".join(str(v) for v in val) if val else " "
            else:
                doc_text = str(val).strip() if val else " "
            
            if not doc_text:
                doc_text = " "

            vec = get_embeddings([doc_text])[0]
            collection.upsert(
                ids=[f"{job['id']}_{field}"],
                documents=[doc_text],
                embeddings=[vec],
                metadatas=[{"job_id": job["id"], "field": field, **meta}],
            )
            total_ids.append(f"{job['id']}_{field}")
    return len(total_ids)


def search_jobs(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict]:
    query_embedding = get_embeddings([query.strip() or " "])[0]

    kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["distances", "documents", "metadatas"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id":       doc_id,
            "score":    1 - results["distances"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


def collection_count(collection: chromadb.Collection) -> int:
    return collection.count()


def get_existing_ids(collection: chromadb.Collection) -> set:
    result = collection.get(include=[])
    return set(result["ids"])