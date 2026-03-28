from __future__ import annotations
import re
import json
import time

from groq import Groq, APIConnectionError, RateLimitError

from config import (
    GROQ_API_KEY_EXTRACT, GROQ_MODEL, EXTRACTION_PROMPT,
    VALID_ROLES, VALID_SENIORITY,
)


def extract_with_groq(title: str, description: str) -> dict:
    """Send a single job to Groq and return a structured extraction dict."""
    if not description or description == "N/A":
        return _empty_extraction()

    description = description[:2000]
    client      = Groq(api_key=GROQ_API_KEY_EXTRACT)

    for attempt in range(3):
        try:
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
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model wraps in ```json ... ```
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            # Extract first JSON object if there is extra surrounding text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            data = json.loads(raw)
            return _validate_extraction(data)

        except json.JSONDecodeError as e:
            print(f"  [Groq] JSON parse error (attempt {attempt+1}/3): {e} | raw: {raw[:200]}")
            if attempt == 2:
                return _empty_extraction()
            time.sleep(1)

        except RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  [Groq] Rate limit hit — waiting {wait}s (attempt {attempt+1}/3)...")
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


# ── helpers ───────────────────────────────────────────────────────────────

def _empty_extraction() -> dict:
    return {
        "role":               "Other",
        "seniority":          "Not specified",
        "description":        None,
        "requirements":       None,
        "experience":         None,
        "skills_must":        [],
        "skills_nice":        [],
        "past_experience":    [],
        "tools_technologies": [],
    }


def _validate_extraction(data: dict) -> dict:
    empty        = _empty_extraction()
    result       = {}
    array_fields = {"skills_must", "skills_nice", "past_experience", "tools_technologies"}
    int_fields   = {"experience"}

    for key, default in empty.items():
        val = data.get(key, default)

        if key in array_fields:
            result[key] = [str(v) for v in val if v] if isinstance(val, list) else []

        elif key in int_fields:
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

    return result