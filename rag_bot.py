from groq import Groq
from chroma_store import search_jobs, init_chroma
from config import GROQ_MODEL, GROQ_API_KEY_CHAT

collection = init_chroma()


def generate_answer(query: str, context: str) -> str:
    client = Groq(api_key=GROQ_API_KEY_CHAT)

    prompt = (
        "You are a helpful AI assistant specialized in the high-tech job market.\n"
        "Use the following job postings to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer concisely and clearly."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()


def build_context(retrieved_jobs) -> str:
    parts = []
    seen_jobs = set()

    for job in retrieved_jobs:
        meta = job.get("metadata", {})
        job_id = meta.get("job_id", job.get("id", "N/A"))
        field_name = meta.get("field", "unknown")

        if (job_id, field_name) in seen_jobs:
            continue

        seen_jobs.add((job_id, field_name))

        parts.append(
            f"{field_name}: {job.get('document', '')}\n"
            f"Job ID: {job_id}\n-----"
        )

    return "\n".join(parts)


def rag_bot(user_query: str, top_k: int = 20) -> str:
    retrieved_jobs = search_jobs(collection, user_query, n_results=top_k)

    if not retrieved_jobs:
        return "No relevant job postings found in the database."

    context = build_context(retrieved_jobs)
    return generate_answer(user_query, context)


if __name__ == "__main__":
    while True:
        question = input("Ask a job-related question: ")
        print("\nGenerating answer...\n")
        print(rag_bot(question))