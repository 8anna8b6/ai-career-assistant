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
DATE_FILTER = "r2592000" #"r86400" last 24 hours,"r604800"last 7 days,"r2592000" last 30 days ,"r7776000" last 90 days

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
  "role": "ONE OF: Software Development|Frontend|Backend|Fullstack|AI / ML|Data Scientist|Data Engineer|Data Analyst|BI|DevOps / Cloud|Mobile|QA / Automation|Security|Embedded / Firmware|Database|Network|System Engineer|Solutions Architect|Team Lead|R&D",
  "seniority": "ONE OF: Intern|Junior|Mid|Senior|Lead|Staff|Principal|Manager|Director|VP|Not specified",
  "description": "4-5 sentences about daily work, systems, team context, impact",
  "experience": <integer years or null>,
  "skills_must": ["all required technologies, tools, frameworks, languages, platforms, databases, cloud services, and methodologies that are NOT marked as Advantage/Preferred/Bonus/Nice to have"],
  "skills_nice": ["only skills explicitly marked as Advantage/Preferred/Bonus/Nice to have"],
  "past_experience": ["development domains, job titles, or industry verticals that the post explicitly requires experience in — e.g. 'Backend development', 'Frontend development', 'Mobile development', 'Fullstack Developer', 'FinTech', 'SaaS', 'embedded systems'. Extract the domain/title from phrases like '3-5 years of experience in X', '4+ years of X development', 'background in X'. Do NOT include generic phrases like 'modern technologies' or 'software development'"]

}
 
Rules:
- role: derive from job TITLE only, not description
- experience: integer years or null. For "X+ years" use X (e.g. "8+ years" -> 8). For a range "X-Y years" use the lower bound X. Never return a value lower than what is written. Search the ENTIRE post for any mention of years of experience
- seniority: derive PRIMARILY from years of experience using these rules: 0-3 years = Junior, 3-5 years = Mid, 5+ years = Senior. Override with title only for explicit leadership roles: Lead/Staff/Principal/Manager/Director/VP/Senior/Mid/Junior. If no experience is mentioned anywhere AND the title has no seniority signal, use "Not specified"
- skills_must: scan the ENTIRE job post for required skills. Include every technology, tool, language, framework, platform, database, cloud service, and methodology that a candidate MUST have. Do not limit to a specific section — many posts mix requirements throughout. Exclude only items explicitly labeled Advantage/Preferred/Bonus/Nice to have
- skills_nice: only items explicitly labeled Advantage/Preferred/Bonus/Nice to have anywhere in the post
- past_experience: extract the domain or title from ANY phrase like "X years of experience in Y", "Y years of Y development", "background in Y", "experience as a Y", or "worked as Y". Capture the Y part (e.g. "Backend development", "Mobile", "FinTech"). Also include explicit job titles (e.g. "Fullstack Developer") or industry verticals (e.g. "SaaS", "FinTech") stated as desired background. Leave [] only if the post mentions zero specific domains or titles
- Always respond in English regardless of input language
- If a field is not mentioned return null for strings/numbers or [] for arrays
- skills_must should NEVER be empty if the job description mentions any technologies — always extract them
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