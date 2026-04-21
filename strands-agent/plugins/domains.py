# Copyright (c) 2025 deep-search-portal
# This source code is licensed under the Apache 2.0 License.

"""Query domain classification and tool-to-domain mapping.

Classifies user queries into research domains and maps each domain to
the tools most likely to produce high-quality results. Used by
ToolRouterPlugin (pre-invocation guidance) and ToolAuditPlugin
(post-invocation verification).

The classifier uses keyword/pattern matching — no LLM call needed.
Domains are not mutually exclusive: a query about "clinical trial
results for trenbolone" matches both ACADEMIC and PRACTITIONER.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Domain definitions ────────────────────────────────────────────────

ACADEMIC = "academic"
PRACTITIONER = "practitioner"
GOVERNMENT = "government"
FINANCIAL = "financial"
YOUTUBE = "youtube"
OSINT = "osint"
FORUM = "forum"
PREPRINT = "preprint"
GENERAL = "general"

ALL_DOMAINS = [
    ACADEMIC, PRACTITIONER, GOVERNMENT, FINANCIAL,
    YOUTUBE, OSINT, FORUM, PREPRINT, GENERAL,
]


@dataclass(frozen=True)
class DomainMatch:
    """Result of classifying a query into research domains."""

    domains: tuple[str, ...]
    """Matched domains, ordered by relevance (most relevant first)."""

    primary: str
    """Single most relevant domain."""

    confidence: float = 1.0
    """Classification confidence (0.0-1.0). Currently always 1.0 for
    keyword matching; reserved for future LLM-based classification."""


# ── Keyword patterns per domain ───────────────────────────────────────
# Each pattern is compiled once at import time. Patterns are checked
# case-insensitively against the full query text.

_DOMAIN_PATTERNS: dict[str, list[re.Pattern]] = {
    ACADEMIC: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:paper|papers|study|studies|journal|doi|pubmed|pmid)\b",
            r"\b(?:citation|citations|peer.review|meta.analysis|systematic.review)\b",
            r"\b(?:research\s+(?:paper|article|finding|evidence))\b",
            r"\b(?:scholar|scholarly|academic|literature\s+review)\b",
            r"\b(?:molecular|mechanism|pathway|receptor|agonist|antagonist)\b",
            r"\b(?:pharmacokinetics|pharmacodynamics|bioavailability|half.life)\b",
            r"\b(?:p.value|confidence.interval|cohort|randomized|placebo)\b",
            r"\b(?:openalex|semantic.scholar|arxiv|crossref)\b",
        ]
    ],
    PRACTITIONER: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:protocol|cycle|stack|dosage|dosing|dose)\b",
            r"\b(?:bloodwork|blood\s*work|blood\s*panel|lab\s*results?)\b",
            r"\b(?:side\s*effect|sides|gyno|acne|hair\s*loss)\b",
            r"\b(?:pct|post.cycle|ai\s+(?:dosing|protocol)|aromatase)\b",
            r"\b(?:trt|hrt|testosterone\s+replacement)\b",
            r"\b(?:steroid|anabolic|sarm|peptide|hgh|igf)\b",
            r"\b(?:tren|deca|anavar|winstrol|dianabol|anadrol|primobolan)\b",
            r"\b(?:mk.677|rad.140|lgd.4033|ostarine|cardarine)\b",
            r"\b(?:bpc.157|tb.500|ipamorelin|cjc.1295|ghrp)\b",
            r"\b(?:bodybuilding|powerlifting|physique|recomp|bulk|cut)\b",
            r"\b(?:supplement|nootropic|biohack)\b",
        ]
    ],
    GOVERNMENT: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:clinical\s*trial|nct\d|clinicaltrials\.gov)\b",
            r"\b(?:fda|ema|who|cdc|nih)\b",
            r"\b(?:adverse\s*event|faers|recall|drug\s*safety)\b",
            r"\b(?:court|lawsuit|litigation|legal|ruling|opinion)\b",
            r"\b(?:sec\s+filing|10.k|10.q|8.k|edgar|proxy\s+statement)\b",
            r"\b(?:offshore|shell\s+company|beneficial\s+owner|icij)\b",
            r"\b(?:regulation|regulatory|compliance|enforcement)\b",
        ]
    ],
    FINANCIAL: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:stock|equity|share\s+price|market\s+cap)\b",
            r"\b(?:revenue|earnings|profit|loss|ebitda|margin)\b",
            r"\b(?:patent|intellectual\s+property|ip\s+portfolio)\b",
            r"\b(?:startup|funding|series\s+[a-e]|venture|ipo)\b",
            r"\b(?:quarterly|annual\s+report|investor|dividend)\b",
            r"\b(?:sec\s+filing|10.k|10.q|proxy|edgar)\b",
        ]
    ],
    YOUTUBE: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:youtube|video|channel|creator|youtuber)\b",
            r"\b(?:transcript|podcast|episode|interview)\b",
            r"\b(?:watch|stream|vlog|content\s+creator)\b",
        ]
    ],
    OSINT: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:osint|open.source.intelligence)\b",
            r"\b(?:censored|censorship|blocked|banned|removed)\b",
            r"\b(?:archived|wayback|web\s*archive|cache)\b",
            r"\b(?:dark\s*web|onion|tor|hidden\s*service)\b",
            r"\b(?:leak|leaked|breach|exposed|whistleblow)\b",
            r"\b(?:propaganda|disinformation|misinformation)\b",
            r"\b(?:surveillance|tracking|monitoring)\b",
        ]
    ],
    FORUM: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:forum|thread|post|user\s+report|experience\s+report)\b",
            r"\b(?:meso.?rx|elite.?fitness|professional.?muscle)\b",
            r"\b(?:anabolic.?minds|t.nation|think.?steroids)\b",
            r"\b(?:underground|source\s+review|vendor\s+review)\b",
            r"\b(?:reddit|subreddit|r/)\b",
            r"\b(?:community|discussion|anecdot)\b",
        ]
    ],
    PREPRINT: [
        re.compile(p, re.IGNORECASE) for p in [
            r"\b(?:preprint|pre.print|biorxiv|medrxiv|chemrxiv)\b",
            r"\b(?:ssrn|osf|working\s+paper)\b",
            r"\b(?:not\s+yet\s+(?:published|peer.reviewed))\b",
            r"\b(?:iacr|eprint|cryptography)\b",
        ]
    ],
}


# ── Tool-to-domain mapping ────────────────────────────────────────────
# Maps each domain to the tool names most relevant for that domain.
# Tool names must match what appears in the Strands tool registry
# (function names for native tools, MCP tool names for MCP tools).

DOMAIN_TOOLS: dict[str, list[str]] = {
    ACADEMIC: [
        # Native knowledge tools
        "openalex_search", "openalex_get_work", "openalex_citation_network",
        "search_pubmed", "pubmed_get_abstract",
        "semantic_scholar_search", "semantic_scholar_recommend",
        "search_google_scholar",
        # Native document tools
        "search_open_access", "download_paper", "resolve_doi_metadata",
        "search_core", "search_springer", "search_zenodo",
        # MCP academic
        "ss_search_papers", "ss_get_paper", "ss_get_paper_citations",
        "ss_get_paper_references", "ss_get_recommendations",
        "arxiv_search_papers", "arxiv_get_paper",
        # Integrity
        "check_retraction", "batch_check_retractions", "search_retractions",
        # Wikidata for entity disambiguation
        "wikidata_search", "wikidata_sparql",
    ],
    PRACTITIONER: [
        # Forum tools
        "forum_search", "forum_read_thread", "forum_deep_dive",
        # YouTube — practitioners share protocols on video
        "search_youtube", "youtube_download_transcript",
        "youtube_search_transcripts", "search_channel_videos",
        # Community
        "reddit_search", "reddit_get_subreddit_posts",
        # Deep research (uncensored)
        "perplexity_deep_research", "grok_deep_research",
        # Web search
        "duckduckgo_search", "brave_web_search",
    ],
    GOVERNMENT: [
        "search_clinical_trials", "get_trial_results",
        "search_fda_adverse_events", "search_fda_recalls",
        "search_court_opinions",
        "search_sec_filings",
        "search_offshore_leaks",
    ],
    FINANCIAL: [
        "search_sec_filings",
        "search_offshore_leaks",
        # Web search for financial news
        "brave_web_search", "brave_news_search",
        "kagi_search", "kagi_enrich_news",
        "perplexity_deep_research",
    ],
    YOUTUBE: [
        "search_youtube", "youtube_download_transcript",
        "youtube_search", "youtube_channel_search",
        "youtube_search_transcripts", "youtube_video_info",
        "youtube_channel_list", "youtube_bulk_transcribe",
        "youtube_get_comments", "youtube_harvest_channel",
        "search_channel_videos", "get_channel_latest_videos",
        "list_channel_videos",
    ],
    OSINT: [
        "wayback_cdx_search", "wayback_diff",
        "wayback_search", "wayback_fetch", "archive_today_fetch",
        "ipfs_fetch", "search_common_crawl",
        "beacon_censorship_info", "search_iacr_eprint",
        # Deep research for censorship-resistant answers
        "grok_deep_research", "perplexity_deep_research",
        # Bright Data for anti-block scraping
        "bd_scrape_as_markdown",
    ],
    FORUM: [
        "forum_search", "forum_read_thread", "forum_deep_dive", "forum_list",
        "reddit_search", "reddit_get_subreddit_posts", "reddit_get_post_details",
    ],
    PREPRINT: [
        "search_biorxiv", "search_biorxiv_by_doi",
        "search_chemrxiv", "search_ssrn",
        "search_osf_preprints", "list_osf_providers",
        # Often want to cross-reference with published versions
        "search_pubmed", "openalex_search",
        "check_retraction",
    ],
    GENERAL: [
        "duckduckgo_search", "brave_web_search", "stract_search",
        "jina_read_url", "kagi_search",
        "perplexity_deep_research",
    ],
}


# ── Domain-to-skill mapping ──────────────────────────────────────────
# Maps domains to skill names that should be auto-activated.

DOMAIN_SKILLS: dict[str, str | None] = {
    ACADEMIC: "academic-research",
    PRACTITIONER: "osint-censored-discovery",
    GOVERNMENT: "government-data",
    FINANCIAL: "financial-research",
    YOUTUBE: "osint-censored-discovery",
    OSINT: "osint-censored-discovery",
    FORUM: "forum-mining",
    PREPRINT: "preprint-pipeline",
    GENERAL: None,
}


# ── Domain-specific guidance text ─────────────────────────────────────
# Injected into the system prompt by ToolRouterPlugin when a domain
# is detected. Concise, actionable, tool-specific.

DOMAIN_GUIDANCE: dict[str, str] = {
    ACADEMIC: (
        "FOR THIS QUERY, prioritize ACADEMIC tools:\n"
        "- OpenAlex (openalex_search, openalex_citation_network) — 240M+ works, citation graphs\n"
        "- Semantic Scholar (ss_search_papers, ss_get_paper_citations) — 200M+ papers\n"
        "- PubMed (search_pubmed, pubmed_get_abstract) — biomedical literature\n"
        "- Google Scholar (search_google_scholar) — broad academic coverage\n"
        "- Preprint servers (search_biorxiv, search_chemrxiv, search_ssrn) — cutting-edge\n"
        "- Document acquisition (download_paper, search_open_access) — full text\n"
        "- Citation integrity (check_retraction) — verify papers aren't retracted\n"
        "START with semantic_scholar or openalex for the core literature, then expand."
    ),
    PRACTITIONER: (
        "FOR THIS QUERY, prioritize PRACTITIONER tools:\n"
        "- Forums (forum_search, forum_deep_dive) — real-world protocols from 14 forums\n"
        "- YouTube (search_youtube, youtube_download_transcript) — practitioner videos\n"
        "- Reddit (reddit_search) — community discussions and experience reports\n"
        "- Deep research (perplexity_deep_research, grok_deep_research) — uncensored\n"
        "START with forum_search for practitioner knowledge, then cross-reference with "
        "YouTube transcripts for video-based protocols."
    ),
    GOVERNMENT: (
        "FOR THIS QUERY, prioritize GOVERNMENT tools:\n"
        "- Clinical trials (search_clinical_trials, get_trial_results) — ClinicalTrials.gov\n"
        "- FDA data (search_fda_adverse_events, search_fda_recalls) — safety signals\n"
        "- Court records (search_court_opinions) — legal rulings and opinions\n"
        "- SEC filings (search_sec_filings) — corporate disclosures, 10-K, 10-Q\n"
        "- Offshore leaks (search_offshore_leaks) — ICIJ database\n"
        "START with the most specific government database for the query."
    ),
    FINANCIAL: (
        "FOR THIS QUERY, prioritize FINANCIAL tools:\n"
        "- SEC filings (search_sec_filings) — 10-K, 10-Q, 8-K, proxy statements\n"
        "- Offshore leaks (search_offshore_leaks) — beneficial ownership, shell companies\n"
        "- News search (brave_news_search, kagi_enrich_news) — financial news\n"
        "- Deep research (perplexity_deep_research) — market analysis\n"
        "START with SEC filings for corporate data, then expand to news sources."
    ),
    YOUTUBE: (
        "FOR THIS QUERY, prioritize YOUTUBE tools:\n"
        "- Search (search_youtube, youtube_search) — find relevant videos\n"
        "- Transcripts (youtube_download_transcript, youtube_search_transcripts) — content analysis\n"
        "- Channel analysis (youtube_channel_search, youtube_channel_list) — creator profiles\n"
        "- Bulk analysis (youtube_bulk_transcribe, youtube_harvest_channel) — deep dives\n"
        "- Comments (youtube_get_comments) — audience reactions and data\n"
        "Use the Breadth→Depth→Quality pipeline: search_youtube for breadth, "
        "search_channel_videos for depth, youtube_download_transcript for quality assessment."
    ),
    OSINT: (
        "FOR THIS QUERY, prioritize OSINT tools:\n"
        "- Web archives (wayback_cdx_search, wayback_diff, archive_today_fetch) — historical\n"
        "- IPFS (ipfs_fetch) — censorship-resistant content\n"
        "- Common Crawl (search_common_crawl) — web-scale text corpus\n"
        "- Censorship detection (beacon_censorship_info) — what's being blocked\n"
        "- Deep research (grok_deep_research) — less censored than alternatives\n"
        "- Bright Data (bd_scrape_as_markdown) — bypass blocks and CAPTCHAs\n"
        "START with wayback/archive tools for removed content, then use uncensored search."
    ),
    FORUM: (
        "FOR THIS QUERY, prioritize FORUM tools:\n"
        "- Forum search (forum_search) — search across 14 forums in 6 languages\n"
        "- Thread reading (forum_read_thread) — full thread content\n"
        "- Deep dive (forum_deep_dive) — exhaustive forum analysis on a topic\n"
        "- Reddit (reddit_search, reddit_get_subreddit_posts) — community discussions\n"
        "START with forum_search across all forums, then use forum_deep_dive on "
        "the most relevant forums. Cross-reference with reddit_search."
    ),
    PREPRINT: (
        "FOR THIS QUERY, prioritize PREPRINT tools:\n"
        "- bioRxiv/medRxiv (search_biorxiv) — biology and medicine preprints\n"
        "- ChemRxiv (search_chemrxiv) — chemistry preprints\n"
        "- SSRN (search_ssrn) — social science and economics\n"
        "- OSF Preprints (search_osf_preprints) — multi-discipline\n"
        "- Cross-reference (search_pubmed, check_retraction) — verify status\n"
        "START with the relevant preprint server, then check if published versions exist."
    ),
    GENERAL: (
        "Use a BROAD search strategy:\n"
        "- Start with duckduckgo_search and brave_web_search for initial coverage\n"
        "- Use jina_read_url to extract full content from promising results\n"
        "- Use perplexity_deep_research for synthesis and gap-filling\n"
        "- Store findings via store_finding as you go"
    ),
}


# ── Classification function ───────────────────────────────────────────


def classify_query(query: str) -> DomainMatch:
    """Classify a query into one or more research domains.

    Uses keyword/pattern matching against the query text. Multiple
    domains can match — the result includes all matches ordered by
    the number of pattern hits (most hits = most relevant).

    Args:
        query: The user's research query text.

    Returns:
        DomainMatch with matched domains and primary domain.
    """
    scores: dict[str, int] = {}
    for domain, patterns in _DOMAIN_PATTERNS.items():
        hits = sum(1 for p in patterns if p.search(query))
        if hits > 0:
            scores[domain] = hits

    if not scores:
        return DomainMatch(domains=(GENERAL,), primary=GENERAL)

    # Sort by hit count descending
    ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return DomainMatch(
        domains=tuple(ranked),
        primary=ranked[0],
    )
