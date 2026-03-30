from __future__ import annotations
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from embeddings import get_embeddings
from utils import build_chroma_metadata


IMPORTANT_FIELDS = [
    "skills_must",
    "skills_nice",
    "description",
    "title",
    "role",
    "company",
    "location",
]



def build_full_text(job: dict) -> str:
    def join_list(val):
        if isinstance(val, list):
            return ", ".join(str(v) for v in val if v)
        return str(val) if val else ""

    return f"""
Job Title: {job.get('title', '')}
Role: {job.get('role', '')}
Seniority: {job.get('seniority', '')}
Company: {job.get('company', '')}
Location: {job.get('location', '')}

Required Skills: {join_list(job.get('skills_must', []))}
Nice Skills: {join_list(job.get('skills_nice', []))}

Experience: {job.get('yearsexperience', '')} years

Past Experience: {join_list(job.get('past_experience', []))}

Description:
{job.get('description', '')}
""".strip()



def init_chroma() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[ChromaDB] Collection '{CHROMA_COLLECTION}' | {collection.count()} vectors")
    return collection



def upsert_jobs(collection: chromadb.Collection, jobs: List[dict]) -> int:
    if not jobs:
        return 0

    total = 0

    for job in jobs:
        job_id = job["id"]
        meta = build_chroma_metadata(job)

        #full job vector
        full_text = build_full_text(job)

        vec = get_embeddings(full_text)[0]

        collection.upsert(
            ids=[f"{job_id}_full"],
            documents=[full_text],
            embeddings=[vec],
            metadatas=[{
                "job_id": job_id,
                "type": "full",
                **meta
            }],
        )
        total += 1

        for field in IMPORTANT_FIELDS:
            val = job.get(field, "")

            if isinstance(val, list):
                text = ", ".join(str(v) for v in val if v)
            else:
                text = str(val).strip() if val else ""

            if not text:
                continue

            vec = get_embeddings(text)[0]

            collection.upsert(
                ids=[f"{job_id}_{field}"],
                documents=[text],
                embeddings=[vec],
                metadatas=[{
                    "job_id": job_id,
                    "type": "field",
                    "field": field,
                    **meta
                }],
            )

            total += 1

    return total


def search_jobs(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict]:

    query_embedding = get_embeddings(query.strip() or " ")[0]

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id": doc_id,
            "score": 1 - results["distances"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    # sort best first
    hits.sort(key=lambda x: x["score"], reverse=True)

    return hits




def collection_count(collection: chromadb.Collection) -> int:
    return collection.count()


def get_existing_ids(collection: chromadb.Collection) -> set:
    result = collection.get(include=[])
    return set(result["ids"])