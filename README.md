# Job RAG System

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/job-scraper.git
cd job-scraper
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` file

Create a `.env` file in the root directory:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jobs_db
DB_USER=postgres
DB_PASSWORD=your_password

GROQ_API_KEY_EXTRACT=your_key_here
GROQ_API_KEY_CHAT=your_key_here
GROQ_MODEL=llama-3.1-8b-instant

CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=linkedin_jobs_test
```

### 4. Run the project

```bash
python main.py
```

---

## Notes

* `.env` is required and not included in the repo
* Make sure PostgreSQL is running
* ChromaDB will be created automatically
