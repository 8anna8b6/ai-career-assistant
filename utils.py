import re
from datetime import datetime, timedelta, timezone


def fmt(seconds: float) -> str:
    """Format seconds into a readable string."""
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{seconds:.1f}s"


def parse_posted_date(text: str):
    """Parse a readable posted-date string into a date object."""
    if not text:
        return None
    text = text.strip()
    now  = datetime.now(timezone.utc).replace(tzinfo=None)

    iso = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        try:
            return datetime.strptime(iso.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    num_match = re.search(r"(\d+)", text)
    n = int(num_match.group(1)) if num_match else 1
    t = text.lower()

    if "שעה" in t or "שעות" in t or "hour" in t:
        return (now - timedelta(hours=n)).date()
    if "יום" in t or "ימים" in t or "day" in t:
        return (now - timedelta(days=n)).date()
    if "שבוע" in t or "שבועות" in t or "week" in t:
        return (now - timedelta(weeks=n)).date()
    if "חודש" in t or "חודשים" in t or "month" in t:
        return (now - timedelta(days=n * 30)).date()
    if "עכשיו" in t or "just" in t or "moments" in t:
        return now.date()
    return None


def build_embedding_text(job: dict) -> str:
    parts = []

    for field in ("title", "role", "seniority", "company", "location"):
        val = job.get(field)
        if val and val != "N/A":
            parts.append(str(val))

    for field in ("description", "requirements"):
        val = job.get(field)
        if val:
            parts.append(str(val))

    for field in ("skills_must", "skills_nice", "tools_technologies", "past_experience"):
        val = job.get(field)
        if isinstance(val, list) and val:
            parts.append(", ".join(val))

    return "\n".join(parts)


def build_chroma_metadata(job: dict) -> dict:
    
    
    scalar_fields = (
        "id", "title", "role", "seniority", "company", 
        "location", "url", "experience", "keyword", "source"
    )
    
    
    list_fields = (
        "skills_must", "skills_nice", "past_experience", "tools_technologies"
    )

    meta = {}

    for f in scalar_fields:
        val = job.get(f)
        if val is not None:
            
            if f == "experience":
                meta[f] = int(val)
            else:
                meta[f] = str(val)
        else:
            meta[f] = "" 

    
    for f in list_fields:
        val = job.get(f)
        if isinstance(val, list) and val:
            meta[f] = ", ".join(str(v) for v in val)
        else:
            meta[f] = ""

    
    posted = job.get("posted_at")
    meta["posted_at"] = str(posted) if posted else ""

    return meta