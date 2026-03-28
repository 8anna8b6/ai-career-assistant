

from __future__ import annotations
import re
import time

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import InvalidSessionIdException, WebDriverException

from config import CHROME_VERSION, DATE_FILTER
from utils import parse_posted_date, fmt


# ── driver ────────────────────────────────────────────────────────────────

def build_driver() -> uc.Chrome:
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--lang=he-IL")
    options.add_argument("--accept-lang=he-IL,he;q=0.9,en-US;q=0.8")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process")

    driver = uc.Chrome(options=options, use_subprocess=True, version_main=CHROME_VERSION)
    driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
        "headers": {"Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8"}
    })
    return driver


def is_driver_alive(driver) -> bool:
    try:
        _ = driver.current_url
        return True
    except Exception:
        return False


# ── URL builders ──────────────────────────────────────────────────────────

def search_url(keyword: str, offset: int = 0) -> str:
    base = (
        f"https://il.linkedin.com/jobs/search/"
        f"?keywords={keyword.replace(' ', '%20')}"
        f"&location=Israel&sortBy=DD&start={offset}"
    )
    if DATE_FILTER:
        base += f"&f_TPR={DATE_FILTER}"
    return base


def api_url(job_id: str) -> str:
    return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


# ── page helpers ──────────────────────────────────────────────────────────

def dismiss_popup(driver) -> None:
    selectors = [
        "button[data-tracking-control-name='public_jobs_contextual-sign-in-modal_modal_dismiss']",
        "button.contextual-sign-in-modal__modal-dismiss",
        "button[aria-label='ביטול']",
        "button.modal__dismiss",
    ]
    for sel in selectors:
        try:
            btn = WebDriverWait(driver, 1).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, sel))
            )
            driver.execute_script("arguments[0].click();", btn)
            return
        except Exception:
            continue


def wait_for_cards(driver, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        cards = driver.find_elements(By.CSS_SELECTOR, "ul.jobs-search__results-list li")
        if any(_is_real_card(c) for c in cards):
            return True
        src = driver.page_source.lower()
        if "authwall" in src or "checkpoint" in src:
            input("  Auth wall — solve manually then press ENTER...")
            return True
        time.sleep(0.5)
    return False


def _is_real_card(card) -> bool:
    try:
        title = card.find_element(By.CSS_SELECTOR, "h3.base-search-card__title").text.strip()
        if not title:
            return False
        card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
        return True
    except Exception:
        return False


def extract_job_id(url: str) -> str | None:
    match = re.search(r"-(\d{7,})", url)
    if match:
        return match.group(1)
    match = re.search(r"(\d{10,})", url)
    return match.group(1) if match else None


def scroll_to_load_all(driver) -> None:
    last_count = 0
    for _ in range(10):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)
        cards = driver.find_elements(By.CSS_SELECTOR, "ul.jobs-search__results-list li")
        if len(cards) == last_count:
            break
        last_count = len(cards)


def fetch_stubs(driver, seen_ids: set) -> list:
    """Extract job card stubs from the current search results page."""
    try:
        WebDriverWait(driver, 8).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "ul.jobs-search__results-list li")
            )
        )
    except Exception:
        return []

    stubs = []
    for card in driver.find_elements(By.CSS_SELECTOR, "ul.jobs-search__results-list li"):
        try:
            title = card.find_element(By.CSS_SELECTOR, "h3.base-search-card__title").text.strip()
            if not title:
                continue
            link_el = card.find_element(By.CSS_SELECTOR, "a.base-card__full-link")
            job_url = link_el.get_attribute("href") or ""
            job_id  = extract_job_id(job_url)

            if not job_id or job_id in seen_ids:
                if job_id in seen_ids:
                    print(f"  Dup: {title}")
                continue

            try:
                company = card.find_element(
                    By.CSS_SELECTOR, "h4.base-search-card__subtitle a.hidden-nested-link"
                ).text.strip() or card.find_element(
                    By.CSS_SELECTOR, "h4.base-search-card__subtitle"
                ).text.strip()
            except Exception:
                company = "N/A"

            try:
                location = card.find_element(
                    By.CSS_SELECTOR, "span.job-search-card__location"
                ).text.strip()
            except Exception:
                location = "N/A"

            stubs.append({
                "id": job_id, "title": title, "company": company,
                "location": location, "url": job_url.split("?")[0],
            })
        except Exception:
            continue

    return stubs


def get_description(driver, job_id: str) -> tuple[str, object, float]:
    """
    Fetch the full job description from the LinkedIn API page.

    Returns (description_text, posted_at_date, elapsed_seconds).
    """
    t0          = time.time()
    MAX_RETRIES = 3

    for attempt in range(MAX_RETRIES):
        try:
            driver.get(api_url(job_id))

            # Check for auth walls
            authwall_start = time.time()
            while time.time() - authwall_start < 3:
                url = driver.current_url
                if "authwall" in url or "login" in url or "checkpoint" in url:
                    wait_secs = 20 + (attempt * 10)
                    print(f"  Auth wall — waiting {wait_secs}s (retry {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait_secs)
                    break
                time.sleep(0.3)

            description = "N/A"
            posted_at   = None

            for sel in [
                "div.show-more-less-html__markup",
                "div.description__text",
                "section.description",
                "div.core-section-container__content",
            ]:
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                    )
                    txt = driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                    if len(txt) > 50:
                        description = txt
                        break
                except Exception:
                    continue

            # Parse posted date
            try:
                time_el = driver.find_element(
                    By.CSS_SELECTOR, "time.posted-time-ago__text, time[datetime]"
                )
                datetime_attr = time_el.get_attribute("datetime")
                posted_at = (
                    parse_posted_date(datetime_attr) if datetime_attr
                    else parse_posted_date(time_el.text)
                )
            except Exception:
                pass

            if not posted_at:
                try:
                    span = driver.find_element(
                        By.CSS_SELECTOR,
                        "span.posted-time-ago__text, .topcard__flavor--metadata"
                    )
                    posted_at = parse_posted_date(span.text)
                except Exception:
                    pass

            if description != "N/A":
                return description, posted_at, time.time() - t0

        except Exception:
            pass

    return "N/A", None, time.time() - t0


# ── keyword scraper ───────────────────────────────────────────────────────

def scrape_keyword(driver, keyword: str, seen_ids: set, remaining: int = 50) -> list[dict]:
    """
    Scrape up to *remaining* new job stubs for *keyword* and return them
    as a list of dicts with raw_description and posted_at already fetched.

    This function is pure scraping: it does NOT call Groq or write to any DB.
    The caller (main.py) handles extraction and storage.
    """
    kw_start = time.time()
    print(f"\n── Keyword: '{keyword}' ──────────────────────────────")

    driver.get(search_url(keyword))
    dismiss_popup(driver)

    if not wait_for_cards(driver):
        print("  No cards found.")
        return []

    scroll_to_load_all(driver)

    all_stubs: list[dict] = []
    page      = 0
    MAX_PAGES = 2
    offset    = 0

    while page < MAX_PAGES:
        page_stubs = fetch_stubs(driver, seen_ids)
        page_count = len(page_stubs)
        new_stubs  = [s for s in page_stubs if s["id"] not in {x["id"] for x in all_stubs}]
        all_stubs.extend(new_stubs)
        print(f"  Page {page+1}: {page_count} cards, {len(new_stubs)} new (total: {len(all_stubs)})")

        if page_count == 0:
            break

        offset += page_count
        page   += 1

        driver.get(search_url(keyword, offset))
        dismiss_popup(driver)
        if not wait_for_cards(driver):
            break
        scroll_to_load_all(driver)

        if page_count < 5:
            break

    if not all_stubs:
        print("  No new cards.")
        return []

    all_stubs = all_stubs[:remaining]
    print(f"  Fetching descriptions for {len(all_stubs)} jobs...")

    results   = []
    job_times = []

    for stub in all_stubs:
        if not is_driver_alive(driver):
            raise WebDriverException("Browser closed")

        try:
            raw_desc, posted_at, elapsed = get_description(driver, stub["id"])
            job_times.append(elapsed)
            seen_ids.add(stub["id"])
            print(f"  [fetch] {stub['title']} | {stub['company']} | {fmt(elapsed)}")
        except (InvalidSessionIdException, WebDriverException):
            raise
        except Exception as e:
            print(f"  [fetch error] {stub['id']}: {e}")
            continue

        stub["raw_description"] = raw_desc
        stub["posted_at"]       = posted_at
        stub["keyword"]         = keyword
        results.append(stub)

    avg = sum(job_times) / len(job_times) if job_times else 0
    print(f"  ✓ {fmt(time.time() - kw_start)} | {len(results)} fetched | avg: {fmt(avg)}/job")
    return results 