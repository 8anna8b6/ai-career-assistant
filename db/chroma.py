
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from db.embeddings import get_embeddings
from pipeline.utils import build_chroma_metadata

log = logging.getLogger(__name__)

# Fields that get their own dedicated embedding vector in addition to the full-text vector
FIELD_VECTORS = [
    "skills_must",
    "skills_nice",
    "description",
    "title",
    "role",
    "company",
    "location",
]


# ── Init ──────────────────────────────────────────────────────────────────────

def init_chroma() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    log.info("ChromaDB collection '%s' | %d vectors.", CHROMA_COLLECTION, collection.count())
    return collection


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_full_text(job: dict) -> str:
    """Combine all job fields into a single string for the full-text vector."""
    def _join(val):
        return ", ".join(str(v) for v in val if v) if isinstance(val, list) else str(val or "")

    return f"""
Job Title: {job.get('title', '')}
Role: {job.get('role', '')}
Seniority: {job.get('seniority', '')}
Company: {job.get('company', '')}
Location: {job.get('location', '')}

Required Skills: {_join(job.get('skills_must', []))}
Nice Skills: {_join(job.get('skills_nice', []))}
Experience: {job.get('yearsexperience', '')} years
Past Experience: {_join(job.get('past_experience', []))}

Description:
{job.get('description', '')}
""".strip()


def _field_text(job: dict, field: str) -> str:
    """Return a clean string for a single job field."""
    val = job.get(field, "")
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val).strip() if val else ""


# ── Writes ────────────────────────────────────────────────────────────────────

def upsert_jobs(collection: chromadb.Collection, jobs: List[dict]) -> int:
    """
    Upsert jobs into ChromaDB.
    Each job produces:
      - 1 full-text vector
      - 1 vector per FIELD_VECTORS field (if the field has content)
    Returns total number of vectors upserted.
    """
    if not jobs:
        return 0

    total = 0

    for job in jobs:
        job_id = job["id"]
        meta   = build_chroma_metadata(job)

        # Build all texts for this job at once
        full_text   = _build_full_text(job)
        field_texts = {f: _field_text(job, f) for f in FIELD_VECTORS}
        non_empty   = {f: t for f, t in field_texts.items() if t}

        # One batch embedding call per job
        all_texts = [full_text] + list(non_empty.values())
        all_vecs  = get_embeddings(all_texts)

        # Full-text vector
        collection.upsert(
            ids=[f"{job_id}_full"],
            documents=[full_text],
            embeddings=[all_vecs[0]],
            metadatas=[{"job_id": job_id, "type": "full", **meta}],
        )
        total += 1

        # Per-field vectors
        for i, (field, text) in enumerate(non_empty.items(), start=1):
            collection.upsert(
                ids=[f"{job_id}_{field}"],
                documents=[text],
                embeddings=[all_vecs[i]],
                metadatas=[{"job_id": job_id, "type": "field", "field": field, **meta}],
            )
            total += 1

    log.info("Upserted %d vectors for %d jobs.", total, len(jobs))
    return total


# ── Reads ─────────────────────────────────────────────────────────────────────

def search_jobs(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 10,
    where: Optional[Dict] = None,
) -> List[Dict]:
    """Semantic search. Returns hits sorted by score descending."""
    query_embedding = get_embeddings(query.strip() or " ")[0]

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = [
        {
            "id":       doc_id,
            "score":    1 - results["distances"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i, doc_id in enumerate(results["ids"][0])
    ]
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


def get_existing_ids(collection: chromadb.Collection) -> set:
    return set(collection.get(include=[])["ids"])


def collection_count(collection: chromadb.Collection) -> int:
    return collection.count()