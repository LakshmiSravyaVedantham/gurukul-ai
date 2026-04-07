"""
web_research.py — Free web research for Gurukul script generation.

Uses DuckDuckGo (no API key) + Wikipedia for topic research.
Facts are fed to Gemma before script generation so episodes are accurate.

Usage:
    python web_research.py "photosynthesis"
    python web_research.py "black holes" --facts 8
    python web_research.py "fractions" --json   # machine-readable output

API:
    from web_research import research_topic
    facts = research_topic("volcanoes")  # returns str summary
"""

import argparse, json, re, sys, textwrap, urllib.parse, urllib.request
from pathlib import Path


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _wikipedia_summary(topic: str, sentences: int = 5) -> str:
    """Fetch the plain-text summary for a topic from Wikipedia's REST API."""
    slug = topic.strip().replace(" ", "_")
    url = (
        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
        + urllib.parse.quote(slug, safe="")
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GurukuIAI/1.0 (educational; kids video pipeline)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        extract = data.get("extract", "")
        # Truncate to `sentences` sentences
        sents = re.split(r"(?<=[.!?])\s+", extract)
        return " ".join(sents[:sentences]).strip()
    except Exception as e:
        return f"[Wikipedia unavailable: {e}]"


def _wikipedia_search(query: str, limit: int = 3) -> list[str]:
    """Search Wikipedia and return titles of top results."""
    params = urllib.parse.urlencode({
        "action": "opensearch",
        "search": query,
        "limit": limit,
        "namespace": 0,
        "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GurukuIAI/1.0 (educational; kids video pipeline)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data[1] if len(data) > 1 else []
    except Exception:
        return []


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """
    DuckDuckGo Instant Answer API + HTML scrape for snippets.
    Returns list of {"title": str, "snippet": str, "url": str}.
    No API key needed.
    """
    results = []

    # 1. Instant Answer API (fast, structured)
    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    })
    url = f"https://api.duckduckgo.com/?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GurukuIAI/1.0 (educational; kids video pipeline)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        # Abstract (top result)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "snippet": data["AbstractText"][:400],
                "url": data.get("AbstractURL", ""),
            })

        # Related topics
        for item in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(item, dict) and item.get("Text"):
                results.append({
                    "title": item.get("Text", "")[:60],
                    "snippet": item.get("Text", "")[:300],
                    "url": item.get("FirstURL", ""),
                })
    except Exception:
        pass

    return results[:max_results]


# ── Kids-safe fact extraction ─────────────────────────────────────────────────

def _make_kid_friendly(text: str) -> str:
    """Strip jargon markers and truncate to one clean sentence."""
    # Remove citations like [1], [note 3]
    text = re.sub(r"\[\w*\d+\w*\]", "", text)
    # Remove parenthetical asides
    text = re.sub(r"\([^)]{0,60}\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Take first sentence only
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return sentences[0].strip() if sentences else text


def _extract_facts(text: str, n: int = 5) -> list[str]:
    """Split text into individual sentences as distinct facts."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    facts = []
    for s in sentences:
        s = s.strip()
        if len(s) > 30:  # skip fragments
            facts.append(s)
        if len(facts) >= n:
            break
    return facts


# ── Main research function ────────────────────────────────────────────────────

def research_topic(topic: str, num_facts: int = 8, verbose: bool = False) -> str:
    """
    Research a topic using DuckDuckGo + Wikipedia.
    Returns a multi-line string of kid-appropriate facts.

    Args:
        topic:     The topic to research (e.g., "photosynthesis")
        num_facts: Number of key facts to collect
        verbose:   Print progress

    Returns:
        A string like:
            KEY FACTS about photosynthesis:
            • Plants use sunlight to make food from water and air.
            • ...
    """
    if verbose:
        print(f"Researching '{topic}'...", flush=True)

    all_facts = []

    # 1. Wikipedia — most reliable for educational content
    if verbose:
        print("  → Wikipedia...", flush=True)
    wiki_text = _wikipedia_summary(topic, sentences=8)
    if not wiki_text.startswith("["):
        all_facts.extend(_extract_facts(wiki_text, n=5))

    # Also try Wikipedia search in case the direct page name differs
    if len(all_facts) < 3:
        titles = _wikipedia_search(topic, limit=2)
        for t in titles:
            if t.lower() != topic.lower():
                extra = _wikipedia_summary(t, sentences=3)
                if not extra.startswith("["):
                    all_facts.extend(_extract_facts(extra, n=2))

    # 2. DuckDuckGo — supplement with web snippets
    if verbose:
        print("  → DuckDuckGo...", flush=True)
    ddg = _ddg_search(f"{topic} for kids facts", max_results=4)
    for r in ddg:
        if r.get("snippet"):
            all_facts.extend(_extract_facts(r["snippet"], n=2))

    # Deduplicate and limit
    seen = set()
    unique_facts = []
    for f in all_facts:
        key = f[:50].lower()
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)
        if len(unique_facts) >= num_facts:
            break

    if not unique_facts:
        return f"[No facts found for '{topic}' — proceeding without research context]"

    lines = [f"KEY FACTS about {topic}:"]
    for fact in unique_facts:
        lines.append(f"• {fact}")

    return "\n".join(lines)


def research_topic_json(topic: str, num_facts: int = 8) -> dict:
    """Like research_topic() but returns structured dict."""
    summary = _wikipedia_summary(topic, sentences=4)
    facts = _extract_facts(_wikipedia_summary(topic, sentences=10), n=num_facts)
    ddg = _ddg_search(f"{topic} facts", max_results=3)

    return {
        "topic": topic,
        "wikipedia_summary": summary,
        "facts": facts,
        "web_snippets": [r["snippet"] for r in ddg if r.get("snippet")],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Free web research for Gurukul topic generation")
    ap.add_argument("topic", help='Topic to research, e.g. "black holes"')
    ap.add_argument("--facts", type=int, default=8, help="Number of facts to collect (default: 8)")
    ap.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    args = ap.parse_args()

    if args.as_json:
        print(json.dumps(research_topic_json(args.topic, args.facts), indent=2))
    else:
        print(research_topic(args.topic, num_facts=args.facts, verbose=True))
