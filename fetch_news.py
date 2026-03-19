#!/usr/bin/env python3
"""
UK News RSS Scraper
Fetches headlines from BBC News, The Guardian, Sky News, Reuters UK, and The Telegraph.
Categorizes stories and outputs to docs/unitedkingdom_news.json
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser
import requests
from dateutil import parser as dateparser
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEEDS = {
    "BBC News":      "https://feeds.bbci.co.uk/news/rss.xml",
    "The Guardian":  "https://www.theguardian.com/uk/rss",
    "Sky News":      "https://feeds.skynews.com/feeds/rss/home.xml",
    "Reuters UK":    "https://feeds.reuters.com/reuters/UKdomesticNews",
    "The Telegraph": "https://www.telegraph.co.uk/news/rss.xml",
}

CATEGORIES   = ["Diplomacy", "Military", "Energy", "Economy", "Local Events"]
MAX_PER_CAT  = 20
MAX_AGE_DAYS = 7
OUTPUT_PATH  = Path("docs/unitedkingdom_news.json")
MODEL        = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------

def fetch_feed(source_name: str, url: str) -> list[dict]:
    """Parse one RSS feed; return list of raw story dicts."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; StratagemDrive/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        if not feed.entries:
            print(f"  [{source_name}] WARNING: feed returned 0 entries (status {resp.status_code})")
            return []

        stories = []
        missing_date = 0
        for entry in feed.entries:
            pub = None
            for date_field in ("published", "updated", "created"):
                raw = getattr(entry, date_field, None)
                if raw:
                    try:
                        pub = dateparser.parse(raw)
                        if pub and pub.tzinfo is None:
                            pub = pub.replace(tzinfo=timezone.utc)
                        break
                    except Exception:
                        continue

            if pub is None:
                pub = datetime.now(timezone.utc)
                missing_date += 1

            stories.append({
                "title":          entry.get("title", "").strip(),
                "url":            entry.get("link",  "").strip(),
                "published_date": pub.isoformat(),
                "source":         source_name,
            })

        print(f"  [{source_name}] fetched {len(stories)} entries "
              f"({missing_date} had no date, defaulted to now)")
        return stories

    except Exception as e:
        print(f"  [{source_name}] ERROR: {e}")
        return []


def within_window(pub_iso: str | None) -> bool:
    """True if the story is no older than MAX_AGE_DAYS."""
    if not pub_iso:
        return False
    try:
        pub = dateparser.parse(pub_iso)
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        return pub >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_existing() -> dict[str, list[dict]]:
    """Load existing JSON, returning a dict keyed by category."""
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for cat in CATEGORIES:
                    data.setdefault(cat, [])
                return data
        except Exception as e:
            print(f"  Could not load existing JSON: {e}")
    return {cat: [] for cat in CATEGORIES}

# ---------------------------------------------------------------------------
# Classification via Claude
# ---------------------------------------------------------------------------

def classify_stories(stories: list[dict], client: Anthropic) -> list[dict]:
    """
    Ask Claude to assign each story a category (or 'DISCARD' if the
    primary subject is not the United Kingdom).
    """
    if not stories:
        return []

    numbered = "\n".join(
        f"{i+1}. {s['title']}" for i, s in enumerate(stories)
    )

    prompt = f"""You are a news classifier. For each headline below, decide:
1. Is the PRIMARY subject the United Kingdom (its government, people, economy, territory, or armed forces)?
   - If NO  → output DISCARD
   - If YES → assign exactly one category from: Diplomacy, Military, Energy, Economy, Local Events

Definitions:
- Diplomacy:    UK foreign policy, international relations, treaties, summits, sanctions
- Military:     UK armed forces, defence procurement, conflicts involving UK troops
- Energy:       UK energy policy, oil/gas, renewables, power grid, fuel prices
- Economy:      UK macroeconomics, trade, inflation, employment, Bank of England, budgets
- Local Events: UK domestic news that does not fit the above (crime, weather, culture, politics, NHS, etc.)

Be GENEROUS with classification — if a story plausibly relates to the UK, keep it.
Reply with ONLY a JSON array, one object per headline, in the same order:
[{{"id":1,"category":"Economy"}},{{"id":2,"category":"DISCARD"}},...]

Headlines:
{numbered}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        print(f"  Claude raw response (first 300 chars): {raw[:300]}")

        classifications = json.loads(raw)

        assigned = 0
        discarded = 0
        for item in classifications:
            idx = item["id"] - 1
            cat = item.get("category", "DISCARD")
            if 0 <= idx < len(stories):
                if cat in CATEGORIES:
                    stories[idx]["category"] = cat
                    assigned += 1
                else:
                    stories[idx]["category"] = "DISCARD"
                    discarded += 1

        print(f"  Classification: {assigned} assigned, {discarded} discarded")

        for s in stories:
            s.setdefault("category", "DISCARD")

    except Exception as e:
        print(f"  Classification error: {e}")
        for s in stories:
            s["category"] = "DISCARD"

    return stories

# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_into(
    existing: dict[str, list[dict]],
    new_stories: list[dict],
) -> dict[str, list[dict]]:
    """Merge new stories into existing store, evicting old entries."""
    known_urls: set[str] = {
        s["url"]
        for cat_list in existing.values()
        for s in cat_list
    }

    for cat in CATEGORIES:
        before = len(existing[cat])
        existing[cat] = [s for s in existing[cat] if within_window(s.get("published_date"))]
        pruned = before - len(existing[cat])
        if pruned:
            print(f"  Pruned {pruned} stale stories from {cat}")

    inserted = 0
    for story in new_stories:
        cat = story.get("category")
        if cat not in CATEGORIES:
            continue
        if story["url"] in known_urls:
            continue
        if not within_window(story.get("published_date")):
            continue

        known_urls.add(story["url"])
        bucket = existing[cat]
        bucket.append(story)
        bucket.sort(key=lambda x: x.get("published_date") or "", reverse=True)
        existing[cat] = bucket[:MAX_PER_CAT]
        inserted += 1

    print(f"  Inserted {inserted} new stories")
    return existing

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== UK News Scraper starting ===")
    print(f"  Run time (UTC): {datetime.now(timezone.utc).isoformat()}")

    client = Anthropic()

    # 1. Fetch all feeds
    all_stories: list[dict] = []
    for source, url in FEEDS.items():
        stories = fetch_feed(source, url)
        all_stories.extend(stories)

    print(f"\nTotal raw stories fetched: {len(all_stories)}")

    if not all_stories:
        print("ERROR: No stories fetched from any feed. Aborting.")
        return

    # 2. Pre-filter by age
    fresh = [s for s in all_stories if within_window(s.get("published_date"))]
    print(f"Fresh stories (within {MAX_AGE_DAYS} days): {len(fresh)}")

    if not fresh:
        print("WARNING: All stories were outside the age window.")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({cat: [] for cat in CATEGORIES}, f, indent=2)
        return

    # 3. Classify in batches of 50
    BATCH = 50
    classified: list[dict] = []
    for i in range(0, len(fresh), BATCH):
        batch = fresh[i: i + BATCH]
        print(f"\n  Classifying batch {i // BATCH + 1} ({len(batch)} stories)...")
        classified.extend(classify_stories(batch, client))

    kept = [s for s in classified if s.get("category") in CATEGORIES]
    print(f"\nStories kept after classification: {len(kept)}")

    # 4. Load existing and merge
    existing = load_existing()
    updated  = merge_into(existing, kept)

    # 5. Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    # 6. Summary
    print("\n=== Output summary ===")
    for cat in CATEGORIES:
        print(f"  {cat}: {len(updated[cat])} stories")
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
