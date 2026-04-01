"""
Trust scoring and serendipity scoring functions.
"""
from __future__ import annotations

import re


# ============================================================================
# Trust Scoring System
# ============================================================================

_TRUST_TIERS: list[tuple[re.Pattern, float]] = [
    # Tier 1: Academic / Government — highest trust
    (re.compile(r'\.edu(/|$)', re.IGNORECASE), 0.9),
    (re.compile(r'\.gov(/|$)', re.IGNORECASE), 0.9),
    (re.compile(r'(arxiv\.org|pubmed|ncbi\.nlm\.nih|doi\.org|springer\.com|nature\.com|science\.org|ieee\.org|acm\.org)', re.IGNORECASE), 0.9),
    (re.compile(r'(sciencedirect\.com|wiley\.com|tandfonline\.com|jstor\.org|researchgate\.net|semanticscholar\.org|scholar\.google)', re.IGNORECASE), 0.85),
    (re.compile(r'(who\.int|ema\.europa\.eu|fda\.gov|nih\.gov|cdc\.gov|europa\.eu)', re.IGNORECASE), 0.9),
    # Tier 2: Major news outlets — high trust
    (re.compile(r'(reuters\.com|apnews\.com|bbc\.co\.uk|bbc\.com|nytimes\.com|washingtonpost\.com|theguardian\.com|economist\.com)', re.IGNORECASE), 0.7),
    (re.compile(r'(cnn\.com|foxnews\.com|nbcnews\.com|abcnews\.go\.com|bloomberg\.com|ft\.com)', re.IGNORECASE), 0.6),
    (re.compile(r'(techcrunch\.com|arstechnica\.com|wired\.com|theverge\.com|vice\.com)', re.IGNORECASE), 0.6),
    # Tier 3: Reference / encyclopedia — moderate-high trust
    (re.compile(r'(wikipedia\.org|wikimedia\.org|wikidata\.org)', re.IGNORECASE), 0.6),
    (re.compile(r'(britannica\.com|investopedia\.com|mayoclinic\.org|webmd\.com|drugs\.com)', re.IGNORECASE), 0.65),
    # Tier 4: Blog / newsletter platforms — moderate trust
    (re.compile(r'(medium\.com|substack\.com|wordpress\.com|blogspot\.com)', re.IGNORECASE), 0.4),
    # Tier 5: Community / discussion — lower trust but high signal for underground topics
    (re.compile(r'(reddit\.com|quora\.com|stackexchange\.com|stackoverflow\.com|news\.ycombinator\.com)', re.IGNORECASE), 0.3),
    (re.compile(r'(4chan\.org|4plebs\.org|archived\.moe|arch\.b4k\.co|warosu\.org)', re.IGNORECASE), 0.2),
    (re.compile(r'(t\.me|telegram\.org|discord\.com|discord\.gg)', re.IGNORECASE), 0.2),
    # Tier 6: Known bodybuilding / steroid forums — domain-specific trust
    (re.compile(r'(eroids\.com|meso-?rx\.com|thinksteroids\.com|evolutionary\.org|steroidology\.com|ugbodybuilding\.com|sfd\.pl)', re.IGNORECASE), 0.35),
    # Tier 7: E-commerce / marketplace — moderate for product sourcing
    (re.compile(r'(amazon\.|ebay\.|allegro\.pl|olx\.pl|alibaba\.com|aliexpress\.com)', re.IGNORECASE), 0.45),
]


def trust_score_url(url: str) -> float:
    """Compute a trust score for a URL based on its domain."""
    if not url:
        return 0.5
    for pattern, score in _TRUST_TIERS:
        if pattern.search(url):
            return score
    return 0.5


# ============================================================================
# Serendipity Scoring
# ============================================================================

def serendipity_score(fact: str, query: str, known_facts: list[str]) -> float:
    """Compute a simplified serendipity score.

    Serendipity = geometric_mean(relevance, novelty, surprise).
    Uses Jaccard-based word overlap as a lightweight proxy for embedding
    similarity (no external embedding model required).
    """
    fact_words = set(fact.lower().split())
    query_words = set(query.lower().split())

    if not fact_words or not query_words:
        return 0.0

    relevance = len(fact_words & query_words) / max(len(fact_words | query_words), 1)
    relevance = min(max(relevance, 0.05), 1.0)

    max_sim = 0.0
    for known in known_facts:
        known_words = set(known.lower().split())
        if not known_words:
            continue
        sim = len(fact_words & known_words) / max(len(fact_words | known_words), 1)
        max_sim = max(max_sim, sim)
    novelty = 1.0 - max_sim
    novelty = min(max(novelty, 0.05), 1.0)

    all_context = query_words.copy()
    for k in known_facts[:10]:
        all_context.update(k.lower().split())
    context_overlap = len(fact_words & all_context) / max(len(fact_words), 1)
    surprise = 1.0 - context_overlap
    surprise = min(max(surprise, 0.05), 1.0)

    return (relevance * novelty * surprise) ** (1.0 / 3.0)
