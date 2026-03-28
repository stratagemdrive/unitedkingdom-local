"""
unitedkingdom_news.py
Fetches UK-focused news from RSS feeds, categorizes stories, and writes
them to docs/unitedkingdom_news.json — capped at 20 per category,
max age 7 days, oldest entries replaced first.
"""

import json
import os
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser
import feedparser
from deep_translator import GoogleTranslator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "docs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "unitedkingdom_news.json")
MAX_PER_CATEGORY = 20
MAX_AGE_DAYS = 7
CATEGORIES = ["Diplomacy", "Military", "Energy", "Economy", "Local Events"]

# RSS feeds — all free, no paywall, UK-focused
FEEDS = [
    # BBC News UK
    {"source": "BBC News", "url": "https://feeds.bbci.co.uk/news/uk/rss.xml"},
    {"source": "BBC News", "url": "https://feeds.bbci.co.uk/news/politics/rss.xml"},
    {"source": "BBC News", "url": "https://feeds.bbci.co.uk/news/business/rss.xml"},
    # The Guardian UK
    {"source": "The Guardian", "url": "https://www.theguardian.com/uk-news/rss"},
    {"source": "The Guardian", "url": "https://www.theguardian.com/politics/rss"},
    {"source": "The Guardian", "url": "https://www.theguardian.com/uk/money/rss"},
    # Sky News
    {"source": "Sky News", "url": "https://feeds.skynews.com/feeds/rss/uk.xml"},
    {"source": "Sky News", "url": "https://feeds.skynews.com/feeds/rss/politics.xml"},
    {"source": "Sky News", "url": "https://feeds.skynews.com/feeds/rss/business.xml"},
    # The Independent
    {"source": "The Independent", "url": "https://www.independent.co.uk/news/uk/rss"},
    {"source": "The Independent", "url": "https://www.independent.co.uk/news/business/rss"},
    # Reuters UK
    {"source": "Reuters", "url": "https://feeds.reuters.com/reuters/UKNews"},
    {"source": "Reuters", "url": "https://feeds.reuters.com/reuters/UKdomesticNews"},
]

# ---------------------------------------------------------------------------
# Category keyword mapping
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "Diplomacy": [
        "diplomacy", "diplomatic", "foreign policy", "embassy", "ambassador",
        "treaty", "bilateral", "multilateral", "nato", "un ", "united nations",
        "foreign minister", "foreign secretary", "summit", "sanctions",
        "international relations", "geopolitical", "eu ", "brexit", "trade deal",
        "commonwealth", "g7", "g20",
    ],
    "Military": [
        "military", "army", "navy", "royal air force", "raf ", "defence",
        "defense", "troops", "soldier", "warfare", "weapons", "missile",
        "nuclear", "armed forces", "mod ", "ministry of defence", "war ",
        "combat", "deployment", "nato", "veteran", "ukraine", "conflict",
        "intelligence", "spy", "mi5", "mi6", "gchq",
    ],
    "Energy": [
        "energy", "oil", "gas", "nuclear power", "renewable", "solar",
        "wind farm", "electricity", "power grid", "net zero", "carbon",
        "climate", "fossil fuel", "north sea", "ofgem", "energy bill",
        "energy price", "heating", "fuel", "coal", "hydrogen", "battery",
        "ev ", "electric vehicle",
    ],
    "Economy": [
        "economy", "economic", "gdp", "inflation", "interest rate",
        "bank of england", "budget", "chancellor", "treasury", "tax",
        "unemployment", "jobs", "recession", "growth", "trade",
        "market", "pound", "sterling", "ftse", "finance", "fiscal",
        "spending", "debt", "deficit", "wage", "cost of living",
        "housing market", "mortgage", "investment", "business",
    ],
    "Local Events": [
        "local", "council", "mayor", "borough", "city", "town", "village",
        "community", "hospital", "school", "crime", "police", "court",
        "murder", "flood", "fire", "transport", "rail", "strike",
        "protest", "election", "nhs", "social care", "planning",
        "housing", "england", "scotland", "wales", "northern ireland",
        "london", "manchester", "birmingham", "leeds", "glasgow",
        "edinburgh", "liverpool", "bristol",
    ],
}


def classify(title: str, description: str) -> str:
    """Return the best-matching category for a story, or None if no match."""
    text = (title + " " + (description or "")).lower()
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else None


def translate_to_english(text: str) -> str:
    """Translate text to English if it is not already English."""
    if not text:
        return text
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated or text
    except Exception as exc:
        log.warning("Translation failed: %s", exc)
        return text


def parse_date(entry) -> datetime | None:
    """Parse a feed entry's published date into a UTC-aware datetime."""
    raw = entry.get("published") or entry.get("updated") or entry.get("created")
    if not raw:
        struct = entry.get("published_parsed") or entry.get("updated_parsed")
        if struct:
            return datetime(*struct[:6], tzinfo=timezone.utc)
        return None
    try:
        dt = dateparser.parse(raw)
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc) if dt else None
    except Exception:
        return None


def fetch_feed(feed_cfg: dict) -> list[dict]:
    """Fetch a single RSS feed and return a list of story dicts."""
    source = feed_cfg["source"]
    url = feed_cfg["url"]
    stories = []
    try:
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            log.warning("Bozo feed (%s): %s", source, url)
            return stories
        cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
        for entry in parsed.entries:
            pub_date = parse_date(entry)
            if pub_date and pub_date < cutoff:
                continue  # too old
            title_raw = entry.get("title", "").strip()
            desc_raw = entry.get("summary", "").strip()
            # Strip HTML tags from description
            desc_clean = re.sub(r"<[^>]+>", "", desc_raw).strip()
            title = translate_to_english(title_raw)
            category = classify(title, desc_clean)
            if not category:
                continue  # not relevant to our categories
            story = {
                "title": title,
                "source": source,
                "url": entry.get("link", ""),
                "published_date": pub_date.isoformat() if pub_date else None,
                "category": category,
            }
            stories.append(story)
    except Exception as exc:
        log.error("Failed to fetch %s (%s): %s", source, url, exc)
    return stories


def load_existing() -> dict:
    """Load the current JSON file, grouped by category."""
    if not os.path.exists(OUTPUT_FILE):
        return {cat: [] for cat in CATEGORIES}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Support both flat list and categorised dict
    if isinstance(data, list):
        grouped = {cat: [] for cat in CATEGORIES}
        for story in data:
            cat = story.get("category")
            if cat in grouped:
                grouped[cat].append(story)
        return grouped
    if isinstance(data, dict) and "stories" in data:
        grouped = {cat: [] for cat in CATEGORIES}
        for story in data["stories"]:
            cat = story.get("category")
            if cat in grouped:
                grouped[cat].append(story)
        return grouped
    return {cat: [] for cat in CATEGORIES}


def merge(existing: dict, fresh: list[dict]) -> dict:
    """
    Merge fresh stories into the existing pool.
    - De-duplicate by URL.
    - Keep max MAX_PER_CATEGORY stories per category.
    - Replace oldest entries first when over the limit.
    - Discard stories older than MAX_AGE_DAYS.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)

    # Index existing by URL
    existing_urls: set[str] = set()
    for stories in existing.values():
        for s in stories:
            existing_urls.add(s["url"])

    # Add only genuinely new stories
    for story in fresh:
        cat = story.get("category")
        if cat not in existing:
            continue
        if story["url"] in existing_urls:
            continue
        existing[cat].append(story)
        existing_urls.add(story["url"])

    # Per-category: drop expired, sort by date desc, cap at MAX_PER_CATEGORY
    for cat in CATEGORIES:
        pool = existing[cat]
        # Drop expired
        pool = [
            s for s in pool
            if s.get("published_date") and
               dateparser.parse(s["published_date"]).astimezone(timezone.utc) >= cutoff
        ]
        # Sort newest-first
        pool.sort(
            key=lambda s: s.get("published_date") or "",
            reverse=True,
        )
        # Cap: keep the MAX_PER_CATEGORY most recent
        existing[cat] = pool[:MAX_PER_CATEGORY]

    return existing


def write_output(grouped: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    flat = []
    for stories in grouped.values():
        flat.extend(stories)
    output = {
        "country": "United Kingdom",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "story_count": len(flat),
        "stories": flat,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    log.info("Wrote %d stories to %s", len(flat), OUTPUT_FILE)


def main():
    log.info("Loading existing data …")
    existing = load_existing()

    log.info("Fetching %d RSS feeds …", len(FEEDS))
    fresh: list[dict] = []
    for cfg in FEEDS:
        results = fetch_feed(cfg)
        log.info("  %s — %d stories from %s", cfg["source"], len(results), cfg["url"])
        fresh.extend(results)
        time.sleep(0.5)  # polite delay

    log.info("Merging %d fresh stories …", len(fresh))
    merged = merge(existing, fresh)

    counts = {cat: len(merged[cat]) for cat in CATEGORIES}
    log.info("Category totals: %s", counts)

    write_output(merged)


if __name__ == "__main__":
    main()
