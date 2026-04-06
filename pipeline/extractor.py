"""
LLM-based job description extraction using Groq.
"""
from __future__ import annotations
import json
import logging
import random
import re
import time
from groq import APIConnectionError, Groq, RateLimitError
from config import (
    EXTRACTION_PROMPT,
    GROQ_API_KEY_CHAT,
    GROQ_API_KEY_EXTRACT,
    GROQ_MODEL,
    VALID_ROLES,
    VALID_SENIORITY,
)

log = logging.getLogger(__name__)

# ── Groq client state ─────────────────────────────────────────────────────────

_GROQ_KEYS = [k for k in [GROQ_API_KEY_EXTRACT, GROQ_API_KEY_CHAT] if k]
_key_index  = 0

# Groq free tier: ~30 req/min per key → enforce a minimum gap between requests
_MIN_GAP_SECONDS   = 3.0
_last_request_time = 0.0


def _get_client() -> Groq:
    return Groq(api_key=_GROQ_KEYS[_key_index % len(_GROQ_KEYS)])


def _rotate_key() -> None:
    global _key_index
    _key_index += 1
    log.info("Groq key rotated → key %d/%d", (_key_index % len(_GROQ_KEYS)) + 1, len(_GROQ_KEYS))


def _throttle() -> None:
    """Sleep if needed to stay under the per-minute rate limit."""
    global _last_request_time
    wait = _MIN_GAP_SECONDS - (time.time() - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.time()


# ── Public API ────────────────────────────────────────────────────────────────

def extract_with_groq(title: str, description: str) -> dict:
    """
    Send a job title + description to Groq and return a structured extraction.
    Retries up to 3 times with key rotation on rate-limit errors.
    """
    if not description or description == "N/A":
        return _empty_extraction()

    client = _get_client()

    for attempt in range(3):
        try:
            _throttle()
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user",   "content": f"Job Title: {title}\n\nJob Description:\n{description[:5000]}\n\nJSON:"},
                ],
                temperature=0,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_and_validate(raw, title)

        except json.JSONDecodeError as e:
            log.warning("Groq JSON parse error (attempt %d/3): %s", attempt + 1, e)
            if attempt == 2:
                return _empty_extraction()
            time.sleep(2)

        except RateLimitError as e:
            err = str(e)
            if "tokens per day" in err or "TPD" in err:
                log.warning("Groq daily token limit reached — skipping job.")
                return _empty_extraction()

            _rotate_key()
            client = _get_client()
            wait   = (2 ** attempt) * 20 + random.uniform(0, 10)
            log.warning("Groq rate limit — rotated key, waiting %.0fs (attempt %d/3).", wait, attempt + 1)
            time.sleep(wait)

        except APIConnectionError as e:
            log.warning("Groq connection error (attempt %d/3): %s", attempt + 1, e)
            if attempt == 2:
                return _empty_extraction()
            time.sleep(5)

        except Exception as e:
            log.warning("Groq unexpected error (attempt %d/3): %s", attempt + 1, e)
            if attempt == 2:
                return _empty_extraction()
            time.sleep(2)

    return _empty_extraction()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_and_validate(raw: str, title: str) -> dict:
    """Strip markdown fences, extract JSON, validate and return."""
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    data = json.loads(raw)
    return _validate(data, title)


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


def _validate(data: dict, title: str) -> dict:
    """Validate and normalise a raw Groq extraction dict."""
    empty  = _empty_extraction()
    result = {}

    for key, default in empty.items():
        # Groq sometimes returns "experience" instead of "yearsexperience"
        llm_key = "experience" if key == "yearsexperience" else key
        val     = data.get(llm_key, data.get(key, default))

        if key in {"skills_must", "skills_nice", "past_experience"}:
            result[key] = [str(v) for v in val if v] if isinstance(val, list) else []

        elif key == "yearsexperience":
            try:
                result[key] = int(val) if val is not None else None
            except (ValueError, TypeError):
                result[key] = None

        elif key == "role":
            result[key] = str(val).strip() if val in VALID_ROLES else _infer_role(title)

        elif key == "seniority":
            result[key] = str(val).strip() if val in VALID_SENIORITY else "Not specified"

        else:
            result[key] = str(val).strip() if val else None

    # Overwrite seniority from years of experience when not a leadership role
    _apply_seniority_heuristic(result)

    return result


def _apply_seniority_heuristic(result: dict) -> None:
    """
    Override seniority based on years of experience for non-leadership roles.
    """
    leadership = {"Lead", "Staff", "Principal", "Manager", "Director", "VP"}
    yrs = result.get("yearsexperience")

    if yrs is None or result.get("seniority") in leadership:
        return

    if yrs <= 3:
        new = "Junior"
    elif yrs <= 5:
        new = "Mid"
    else:
        new = "Senior"

    if new != result.get("seniority"):
        log.debug(
            "Seniority override: '%s' → '%s' (yearsexperience=%d)",
            result.get("seniority"), new, yrs,
        )
    result["seniority"] = new


def _infer_role(title: str) -> str:
    """Keyword-based role fallback when the LLM returns an unrecognised value."""
    t = title.lower()
    checks = [
        (["frontend", "front-end", "front end", "ui developer", "react developer",
          "angular developer", "vue developer", "web developer"],        "Frontend"),
        (["backend", "back-end", "back end", "server-side", "node developer"],   "Backend"),
        (["fullstack", "full-stack", "full stack"],                               "Fullstack"),
        (["machine learning", "deep learning", "nlp", "computer vision",
          "llm", "ai engineer", "ml engineer", "artificial intelligence"],        "AI / ML"),
        (["data scientist"],                                                       "Data Scientist"),
        (["data engineer", "etl", "pipeline engineer"],                           "Data Engineer"),
        (["data analyst", "bi developer", "business intelligence"],               "Data Analyst"),
        (["devops", "cloud engineer", "sre", "site reliability",
          "platform engineer", "infrastructure engineer"],                        "DevOps / Cloud"),
        (["mobile", "android", "ios", "flutter", "react native"],                 "Mobile"),
        (["qa ", "quality engineer", "automation engineer",
          "test engineer", "sdet"],                                               "QA / Automation"),
        (["security", "cyber", "infosec", "penetration", "appsec"],              "Security"),
        (["embedded", "firmware"],                                                 "Embedded / Firmware"),
        (["solutions architect", "solution architect", "system architect"],       "Solutions Architect"),
        (["team lead", "tech lead", "engineering manager"],                       "Team Lead"),
        (["software engineer", "software developer", "programmer",
          "developer", "engineer"],                                               "Software Development"),
    ]

    for keywords, role in checks:
        if any(k in t for k in keywords):
            return role

    # Special case: product manager (needs AND logic)
    if ("product" in t and "manager" in t) or any(
        k in t for k in ["product lead", "vp product", "head of product", "product owner"]
    ):
        return "Product Manager"

    return "Other"