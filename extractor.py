from __future__ import annotations
import re
import json
import time
import random

from groq import Groq, APIConnectionError, RateLimitError

from config import (
    GROQ_API_KEY_EXTRACT, GROQ_API_KEY_CHAT, GROQ_MODEL, EXTRACTION_PROMPT,
    VALID_ROLES, VALID_SENIORITY,
)


_GROQ_KEYS = [k for k in [GROQ_API_KEY_EXTRACT, GROQ_API_KEY_CHAT] if k]
_key_index  = 0


# Groq free tier: 30 req/min per key

_MIN_SECONDS_BETWEEN_REQUESTS = 3.0 
_last_request_time: float = 0.0


def _get_groq_client() -> Groq:
    return Groq(api_key=_GROQ_KEYS[_key_index % len(_GROQ_KEYS)])


def _rotate_key() -> None:
    global _key_index
    _key_index += 1
    print(f"  [Groq] Rotated to key {(_key_index % len(_GROQ_KEYS)) + 1}/{len(_GROQ_KEYS)}")


def _wait_for_rate_limit() -> None:
    """Enforce minimum gap between requests to stay under 30 req/min."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    wait    = _MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.time()




def extract_with_groq(title: str, description: str) -> dict:
    """Send a single job to Groq and return a structured extraction dict."""
    if not description or description == "N/A":
        return _empty_extraction()

    description = description[:5000]
    client = _get_groq_client()

    for attempt in range(3):
        try:
            _wait_for_rate_limit()   

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": f"Job Title: {title}\n\nJob Description:\n{description}\n\nJSON:",
                    },
                ],
                temperature=0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            # Extract first JSON object if there is extra surrounding text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            data = json.loads(raw)
            return _validate_extraction(data, title)

        except json.JSONDecodeError as e:
            print(f"  [Groq] JSON parse error (attempt {attempt+1}/3): {e} | raw: {raw[:200]}")
            if attempt == 2:
                return _empty_extraction()
            time.sleep(2)

        except RateLimitError as e:
            err_msg = str(e)

            # Daily limit hit — no point retrying, return empty
            if "tokens per day" in err_msg or "TPD" in err_msg:
                print(f"  [Groq] Daily token limit reached. Skipping job.")
                return _empty_extraction()

            # Per-minute limit — rotate key + exponential backoff
            _rotate_key()
            client = _get_groq_client()
            wait   = (2 ** attempt) * 20 + random.uniform(0, 10)  # 20s, 40s, 80s + jitter
            print(f"  [Groq] Rate limit (per-min) — rotated key, waiting {wait:.0f}s (attempt {attempt+1}/3)...")
            time.sleep(wait)

        except APIConnectionError as e:
            print(f"  [Groq] Connection error (attempt {attempt+1}/3): {e}")
            if attempt == 2:
                return _empty_extraction()
            time.sleep(5)

        except Exception as e:
            print(f"  [Groq] Error (attempt {attempt+1}/3): {e}")
            if attempt == 2:
                return _empty_extraction()
            time.sleep(2)

    return _empty_extraction()




def _infer_role_from_title(title: str) -> str:
    """Fallback role inference from job title using keyword matching."""
    t = title.lower()
    if any(k in t for k in ["frontend", "front-end", "front end", "ui developer", "ux developer",
                              "web developer", "react developer", "angular developer", "vue developer"]):
        return "Frontend"
    if any(k in t for k in ["backend", "back-end", "back end", "server-side", "api developer",
                              "node developer"]):
        return "Backend"
    if any(k in t for k in ["fullstack", "full-stack", "full stack"]):
        return "Fullstack"
    if any(k in t for k in ["machine learning", "deep learning", "nlp", "computer vision",
                              "llm", "ai engineer", "ml engineer", "artificial intelligence"]):
        return "AI / ML"
    if "data scientist" in t:
        return "Data Scientist"
    if any(k in t for k in ["data engineer", "etl", "pipeline engineer"]):
        return "Data Engineer"
    if any(k in t for k in ["data analyst", "bi developer", "business intelligence"]):
        return "Data Analyst"
    if any(k in t for k in ["devops", "cloud engineer", "sre", "site reliability",
                              "platform engineer", "infrastructure engineer"]):
        return "DevOps / Cloud"
    if any(k in t for k in ["mobile", "android", "ios", "flutter", "react native"]):
        return "Mobile"
    if any(k in t for k in ["qa ", "quality engineer", "automation engineer",
                              "test engineer", "sdet"]):
        return "QA / Automation"
    if any(k in t for k in ["security", "cyber", "infosec", "penetration", "appsec"]):
        return "Security"
    if any(k in t for k in ["embedded", "firmware"]):
        return "Embedded / Firmware"
    if any(k in t for k in ["solutions architect", "solution architect", "system architect"]):
        return "Solutions Architect"
    if ("product" in t and "manager" in t) or \
        any(k in t for k in ["product lead", "vp product", "head of product", "product owner"]):
        return "Product Manager"
    if any(k in t for k in ["team lead", "tech lead", "engineering manager"]):
        return "Team Lead"
    if any(k in t for k in ["software engineer", "software developer", "programmer",
                              "developer", "engineer"]):
        return "Software Development"
    return "Other"




def _empty_extraction() -> dict:
    return {
        "role":            "Other",
        "seniority":       "Not specified",
        "description":     None,
        "yearsexperience": None,
        "skills_must":     [],
        "skills_nice":     [],
        "past_experience": [],
    }


def _validate_extraction(data: dict, title: str = "") -> dict:
    empty        = _empty_extraction()
    result       = {}
    array_fields = {"skills_must", "skills_nice", "past_experience"}

    for key, default in empty.items():
       
        llm_key = "experience" if key == "yearsexperience" else key
        val     = data.get(llm_key, data.get(key, default))

        if key in array_fields:
            result[key] = [str(v) for v in val if v] if isinstance(val, list) else []

        elif key == "yearsexperience":
            try:
                result[key] = int(val) if val is not None else None
            except (ValueError, TypeError):
                result[key] = None

        elif key == "role":
            result[key] = str(val).strip() if val in VALID_ROLES else "Other"

        elif key == "seniority":
            result[key] = str(val).strip() if val in VALID_SENIORITY else "Not specified"

        else:
            result[key] = str(val).strip() if val else None


    if result.get("role") == "Other":
        result["role"] = _infer_role_from_title(title)

    leadership_seniority = {"Lead", "Staff", "Principal", "Manager", "Director", "VP"}
    yrs = result.get("yearsexperience")
    if yrs is not None and result.get("seniority") not in leadership_seniority:
        if yrs <= 3:
            result["seniority"] = "Junior"
        elif yrs <= 5:
            result["seniority"] = "Mid"
        else:
            result["seniority"] = "Senior"

    return result