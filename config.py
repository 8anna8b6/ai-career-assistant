from dotenv import load_dotenv
import os

load_dotenv()

# ───────── Database config ─────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# ───────── Groq config ─────────
GROQ_API_KEY_EXTRACT = os.getenv("GROQ_API_KEY_EXTRACT")
GROQ_API_KEY_CHAT = os.getenv("GROQ_API_KEY_CHAT")
GROQ_MODEL = os.getenv("GROQ_MODEL")

# ───────── Other configs (unchanged) ─────────
EMBEDDING_DIM = 1536
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_DIR")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION")


# ───────── Scraper ─────────
TARGET_JOBS = 20
CHROME_VERSION = 145
DATE_FILTER = "r2592000"

KEYWORDS = [
    "software engineer", "software developer", "fullstack developer", "full stack developer",
    "developer", "software", "programmer", "r&d",
    "backend", "backend developer", "frontend", "frontend developer",
    "fullstack", "full stack", "web", "application", "systems",
    "python", "java", "javascript", "typescript", "go", "ruby", "php", "kotlin",
    "react", "angular", "html", "css",
    "data analyst", "data engineer", "data scientist", "ai", "machine learning",
    "deep learning", "nlp", "computer vision", "big data", "it", "bi",
    "cloud", "aws", "azure", "docker", "kubernetes", "devops", "ci cd",
    "database", "sql", "nosql", "mongodb", "postgres", "mysql",
    "security", "cyber", "infosec", "penetration", "appsec",
    "android", "ios", "mobile", "flutter", "react native",
    "qa", "automation",
    "algorithm", "algorithms", "microservices", "api", "integration", "network", "linux",
    "infrastructure", "platform", "sre", "site reliability",
    "spark", "kafka", "hadoop", "etl", "pipeline",
    "embedded", "firmware",
]
EXTRACTION_PROMPT = """You are a job data extractor. Given a job title and description, return ONLY a valid JSON object. No markdown, no explanation, no extra text.

{
  "role": "ONE OF: Software Development|Frontend|Backend|Fullstack|AI / ML|Data Scientist|Data Engineer|Data Analyst|BI|DevOps / Cloud|Mobile|QA / Automation|Security|Embedded / Firmware|Database|Network|System Engineer|Solutions Architect|Team Lead|R&D|Other",
  "seniority": "ONE OF: Intern|Junior|Mid|Senior|Lead|Staff|Principal|Manager|Director|VP|C-Level|Not specified",
  "description": "4-5 sentences about daily work, systems, team context, impact",
  "requirements": "comma-separated key requirements",
  "experience": <integer years or null>,
  "skills_must": ["required skills"],
  "skills_nice": ["nice-to-have skills"],
  "past_experience": ["relevant past roles or domains"],
  "tools_technologies": ["tools from skills_must only"]
}

Rules:
- role and seniority: derive from job TITLE only, not description
- experience: integer or null — use lower bound of any range (e.g. "3-5 years" -> 3)
- skills_must: only skills marked "required/must/essential"
- skills_nice: only skills marked "advantage/preferred/bonus/nice to have"
- tools_technologies: only items that also appear in skills_must
- past_experience: previous job titles or domains mentioned as relevant background
- Always respond in English regardless of input language
- If a field is not mentioned return null for strings/numbers or [] for arrays
"""


VALID_ROLES = {
    "Software Development", "Frontend", "Backend", "Fullstack", "AI / ML",
    "Data Scientist", "Data Engineer", "Data Analyst", "BI", "DevOps / Cloud",
    "Mobile", "QA / Automation", "Security", "Embedded / Firmware", "Database",
    "Network", "System Engineer", "Solutions Architect", "Team Lead", "R&D", "Other"
}

VALID_SENIORITY = {
    "Intern", "Junior", "Mid", "Senior", "Lead", "Staff",
    "Principal", "Manager", "Director", "VP", "Not specified"
}