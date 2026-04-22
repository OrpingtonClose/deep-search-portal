"""
Comprehensive research tools for the Deep Agents proxy.

Ports the key data-source tools from the Strands agent (MiroThinker) to
LangChain ``@tool`` format so the deepagents LangGraph agent has access to
the same external data sources.

All tools use free public APIs unless noted. API keys are read from env
vars at import time (sourced from /opt/.env on Vast.ai VMs).

Categories:
  1. Web Search        — Brave, Exa, DuckDuckGo (in deepagents_proxy.py)
  2. Academic           — OpenAlex, Semantic Scholar, PubMed, CrossRef
  3. Preprints          — bioRxiv, medRxiv
  4. Government/Legal   — ClinicalTrials.gov, OpenFDA, SEC EDGAR, CourtListener
  5. Research Integrity — Open Retractions, CrossRef retraction metadata
  6. Archives/OSINT     — Wayback Machine CDX
  7. Content Extraction — Jina Reader
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta

import httpx
from langchain_core.tools import tool as langchain_tool

log = logging.getLogger("deepagents-proxy")

_HTTP = httpx.Client(timeout=30, follow_redirects=True)
_UA = "DeepAgentsResearch/1.0 (research agent)"


# ════════════════════════════════════════════════════════════════════════
# 2. ACADEMIC — free APIs, no keys required
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def openalex_search(
    query: str,
    max_results: int = 10,
    filter_oa: bool = False,
) -> str:
    """Search OpenAlex for academic works (240M+ papers, books, datasets). Free, no key.

    Args:
        query: Search query (title, keywords, concepts).
        max_results: Maximum results (default 10, max 50).
        filter_oa: If True, only open-access works.

    Returns:
        Formatted list of works with authors, citations, DOIs, and PDF links.
    """
    params: dict = {
        "search": query,
        "per_page": min(max_results, 50),
        "sort": "relevance_score:desc",
        "select": "id,title,authorships,publication_year,primary_location,"
                  "open_access,cited_by_count,doi,type,is_retracted,"
                  "abstract_inverted_index",
    }
    if filter_oa:
        params["filter"] = "open_access.is_oa:true"

    try:
        resp = _HTTP.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"OpenAlex error: {exc}"

    works = data.get("results", [])
    total = data.get("meta", {}).get("count", 0)
    if not works:
        return f"No OpenAlex results for: {query}"

    out = [f"**OpenAlex: {query}** ({len(works)} of {total:,})\n"]
    for i, w in enumerate(works[:max_results], 1):
        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])[:4]
        )
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        oa = w.get("open_access", {})
        pdf = oa.get("oa_url", "")
        retracted = " RETRACTED" if w.get("is_retracted") else ""

        abstract = ""
        idx = w.get("abstract_inverted_index")
        if idx and isinstance(idx, dict):
            pairs = []
            for word, positions in idx.items():
                for pos in positions:
                    pairs.append((pos, word))
            pairs.sort()
            abstract = " ".join(x[1] for x in pairs)[:250]

        out.append(
            f"{i}. **{w.get('title', '?')}**{retracted}\n"
            f"   {authors} ({w.get('publication_year', '')})\n"
            f"   Cited: {w.get('cited_by_count', 0)}"
            + (f" | DOI: {doi}" if doi else "")
            + (f"\n   PDF: {pdf}" if pdf else "")
            + (f"\n   {abstract}..." if abstract else "")
        )
    return "\n\n".join(out)


@langchain_tool
def semantic_scholar_search(
    query: str,
    max_results: int = 10,
    year_range: str = "",
) -> str:
    """Search Semantic Scholar (200M+ papers, AI-powered relevance). Free.

    Args:
        query: Search query (natural language works well).
        max_results: Maximum results (default 10, max 100).
        year_range: Year filter, e.g. "2020-2024" or "2020-".

    Returns:
        Papers with abstracts, citation counts, and open-access links.
    """
    headers = {"User-Agent": _UA}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    params: dict = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,abstract,citationCount,openAccessPdf,"
                  "externalIds,journal,url",
    }
    if year_range:
        params["year"] = year_range

    try:
        resp = _HTTP.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
        )
        if resp.status_code == 429:
            return "Semantic Scholar rate-limited. Wait a minute and retry."
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"Semantic Scholar error: {exc}"

    papers = data.get("data", [])
    if not papers:
        return f"No Semantic Scholar results for: {query}"

    out = [f"**Semantic Scholar: {query}** ({len(papers)} results)\n"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(
            a.get("name", "") for a in (p.get("authors") or [])[:4]
        )
        ext = p.get("externalIds", {}) or {}
        doi = ext.get("DOI", "")
        pdf = (p.get("openAccessPdf") or {}).get("url", "")
        abstract = (p.get("abstract") or "")[:200]

        out.append(
            f"{i}. **{p.get('title', '?')}**\n"
            f"   {authors} ({p.get('year', '')})\n"
            f"   Cited: {p.get('citationCount', 0)}"
            + (f" | DOI: {doi}" if doi else "")
            + (f"\n   PDF: {pdf}" if pdf else "")
            + (f"\n   {abstract}..." if abstract else "")
        )
    return "\n\n".join(out)


@langchain_tool
def search_pubmed(query: str, max_results: int = 10) -> str:
    """Search PubMed for biomedical literature (36M+ citations). Free, no key.

    Args:
        query: Search query (MeSH terms, drug names, conditions, authors).
        max_results: Maximum results (default 10).

    Returns:
        PubMed articles with titles, authors, abstracts, and PMIDs.
    """
    try:
        # Step 1: search for IDs
        search_resp = _HTTP.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": min(max_results, 50),
                "retmode": "json",
                "sort": "relevance",
            },
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed results for: {query}"

        # Step 2: fetch summaries
        fetch_resp = _HTTP.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
        )
        fetch_resp.raise_for_status()
        result = fetch_resp.json().get("result", {})
    except Exception as exc:
        return f"PubMed error: {exc}"

    out = [f"**PubMed: {query}** ({len(ids)} results)\n"]
    for pmid in ids:
        art = result.get(pmid, {})
        if not isinstance(art, dict):
            continue
        title = art.get("title", "?")
        authors = ", ".join(
            a.get("name", "") for a in art.get("authors", [])[:4]
        )
        source = art.get("source", "")
        pubdate = art.get("pubdate", "")
        doi_list = [
            x.get("value", "")
            for x in art.get("articleids", [])
            if x.get("idtype") == "doi"
        ]
        doi = doi_list[0] if doi_list else ""

        out.append(
            f"- **{title}**\n"
            f"  {authors} | {source} ({pubdate})\n"
            f"  PMID: {pmid}"
            + (f" | DOI: {doi}" if doi else "")
        )
    return "\n\n".join(out)


@langchain_tool
def resolve_doi(doi: str) -> str:
    """Resolve a DOI to full metadata via CrossRef (150M+ works). Free.

    Args:
        doi: The DOI to resolve (e.g. "10.1038/s41586-019-1099-1").

    Returns:
        Paper metadata: title, authors, journal, year, abstract, and links.
    """
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    try:
        resp = _HTTP.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        d = resp.json().get("message", {})
    except Exception as exc:
        return f"CrossRef error for {doi}: {exc}"

    title = " ".join(d.get("title", ["Unknown"]))
    authors = ", ".join(
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in d.get("author", [])[:6]
    )
    year_parts = d.get("published-print", d.get("published-online", {}))
    try:
        year = str((year_parts or {}).get("date-parts", [[""]])[0][0])
    except (IndexError, KeyError):
        year = ""
    journal = " ".join(d.get("container-title", [""]))
    abstract = d.get("abstract", "")[:500]
    url = d.get("URL", f"https://doi.org/{doi}")
    cited = d.get("is-referenced-by-count", 0)

    return (
        f"**{title}**\n"
        f"Authors: {authors}\n"
        f"Journal: {journal} ({year}) | Cited by: {cited}\n"
        f"DOI: {doi}\n"
        f"URL: {url}"
        + (f"\n\nAbstract: {abstract}" if abstract else "")
    )


# ════════════════════════════════════════════════════════════════════════
# 3. PREPRINTS — bioRxiv / medRxiv
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def search_biorxiv(
    query: str,
    server: str = "biorxiv",
    max_results: int = 10,
) -> str:
    """Search bioRxiv or medRxiv for preprints. Free, no key.

    Contains pre-peer-review research including controversial findings,
    negative results, and studies that may never be journal-published.

    Args:
        query: Search query (keywords, author names).
        server: "biorxiv" or "medrxiv" (default: biorxiv).
        max_results: Maximum results (default 10).

    Returns:
        Preprints with metadata and PDF links.
    """
    if server not in ("biorxiv", "medrxiv"):
        return "server must be 'biorxiv' or 'medrxiv'"

    date_from = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = _HTTP.get(
            f"https://api.biorxiv.org/details/{server}/{date_from}/{date_to}/0/{min(max_results * 3, 75)}",
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"{server} API error: {exc}"

    collection = data.get("collection", [])
    if not collection:
        return f"No {server} preprints in the last 90 days matching your query."

    # Filter by query terms
    terms = query.lower().split()
    filtered = []
    for p in collection:
        text = f"{p.get('title', '')} {p.get('abstract', '')} {p.get('authors', '')}".lower()
        if all(t in text for t in terms):
            filtered.append(p)

    if not filtered:
        return f"No {server} preprints matching '{query}' in the last 90 days ({len(collection)} preprints checked)."

    out = [f"**{server} preprints: {query}** ({len(filtered)} matches)\n"]
    for i, p in enumerate(filtered[:max_results], 1):
        doi = p.get("doi", "")
        ver = p.get("version", "1")
        pdf = f"https://www.{server}.org/content/{doi}v{ver}.full.pdf" if doi else ""
        abstract = (p.get("abstract") or "")[:200]

        out.append(
            f"{i}. **{p.get('title', '?')}**\n"
            f"   {(p.get('authors', '') or '')[:120]}\n"
            f"   Date: {p.get('date', '')} | Category: {p.get('category', '')}\n"
            f"   DOI: {doi}"
            + (f"\n   PDF: {pdf}" if pdf else "")
            + (f"\n   {abstract}..." if abstract else "")
        )
    return "\n\n".join(out)


# ════════════════════════════════════════════════════════════════════════
# 4. GOVERNMENT / LEGAL — all free, no keys
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def search_clinical_trials(
    query: str,
    status: str = "",
    max_results: int = 10,
) -> str:
    """Search ClinicalTrials.gov for clinical trials (v2 API). Free, no key.

    Args:
        query: Drug name, condition, sponsor, etc.
        status: Filter: COMPLETED, TERMINATED, SUSPENDED, WITHDRAWN, RECRUITING.
        max_results: Maximum results (default 10).

    Returns:
        Trials with status, sponsor, enrollment, and result availability.
    """
    params: dict = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,"
                  "CompletionDate,LeadSponsorName,EnrollmentCount,"
                  "BriefSummary,HasResults,WhyStopped,Condition,InterventionName",
        "sort": "LastUpdatePostDate:desc",
    }
    if status:
        params["filter.overallStatus"] = status

    try:
        resp = _HTTP.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            headers={"User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"ClinicalTrials.gov error: {exc}"

    studies = data.get("studies", [])
    if not studies:
        return f"No clinical trials for: {query}"

    out = [f"**Clinical trials: {query}** ({len(studies)} results)\n"]
    for i, s in enumerate(studies[:max_results], 1):
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        design = proto.get("designModule", {})
        results_sec = s.get("resultsSection", {})

        nct = ident.get("nctId", "")
        title = ident.get("briefTitle", "?")
        overall = status_mod.get("overallStatus", "")
        sponsor = (sponsor_mod.get("leadSponsor") or {}).get("name", "")
        phases = ", ".join(design.get("phases", []))
        enrollment = (design.get("enrollmentInfo") or {}).get("count", "")
        has_results = "[RESULTS]" if results_sec else "[NO RESULTS]"
        why_stopped = status_mod.get("whyStopped", "")

        out.append(
            f"{i}. **{title}** ({nct})\n"
            f"   Status: {overall} {has_results} | Phase: {phases or 'N/A'}\n"
            f"   Sponsor: {sponsor} | Enrollment: {enrollment}"
            + (f"\n   WHY STOPPED: {why_stopped}" if why_stopped else "")
        )
    return "\n\n".join(out)


@langchain_tool
def search_fda_adverse_events(
    drug_name: str = "",
    reaction: str = "",
    max_results: int = 10,
) -> str:
    """Search FDA Adverse Event Reporting System (FAERS). Free, no key.

    Reports of adverse drug reactions including deaths and hospitalizations.

    Args:
        drug_name: Drug brand or generic name.
        reaction: Adverse reaction (e.g. "death", "liver failure").
        max_results: Maximum results (default 10).

    Returns:
        Adverse event reports with severity, drugs, and reactions.
    """
    terms = []
    if drug_name:
        terms.append(f'patient.drug.medicinalproduct:"{drug_name}"')
    if reaction:
        terms.append(f'patient.reaction.reactionmeddrapt:"{reaction}"')
    if not terms:
        return "Provide at least drug_name or reaction."

    try:
        resp = _HTTP.get(
            "https://api.fda.gov/drug/event.json",
            params={
                "search": " AND ".join(terms),
                "limit": min(max_results, 100),
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"OpenFDA error: {exc}"

    results = data.get("results", [])
    total = data.get("meta", {}).get("results", {}).get("total", 0)
    if not results:
        return f"No FDA adverse events for: {drug_name or reaction}"

    out = [f"**FDA Adverse Events** ({len(results)} of {total:,})\n"]
    for i, ev in enumerate(results[:max_results], 1):
        date = ev.get("receiptdate", "")
        if date and len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        flags = []
        if ev.get("seriousnessdeath"):
            flags.append("DEATH")
        if ev.get("seriousnesslifethreatening"):
            flags.append("LIFE-THREATENING")
        if ev.get("seriousnesshospitalization"):
            flags.append("HOSPITALIZED")
        severity = " | ".join(flags) if flags else "non-serious"

        patient = ev.get("patient", {})
        drugs = [d.get("medicinalproduct", "") for d in patient.get("drug", [])[:5] if d.get("medicinalproduct")]
        reactions = [r.get("reactionmeddrapt", "") for r in patient.get("reaction", [])[:5]]

        out.append(
            f"{i}. **{severity}** ({date})\n"
            f"   Drugs: {', '.join(drugs)}\n"
            f"   Reactions: {', '.join(reactions)}"
        )
    return "\n\n".join(out)


@langchain_tool
def search_sec_filings(
    query: str,
    filing_type: str = "",
    max_results: int = 10,
) -> str:
    """Search SEC EDGAR for corporate filings. Free, no key.

    10-K (annual), 10-Q (quarterly), 8-K (events), 4 (insider trading).

    Args:
        query: Company name, topic, or full-text search.
        filing_type: Filter: "10-K", "10-Q", "8-K", "4", "DEF 14A".
        max_results: Maximum results (default 10).

    Returns:
        SEC filings with entity names, dates, and filing types.
    """
    headers = {
        "User-Agent": "DeepAgentsResearch research@deep-search.uk",
        "Accept": "application/json",
    }

    try:
        resp = _HTTP.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": query,
                "forms": filing_type,
                "from": 0,
                "size": min(max_results, 50),
            },
            headers=headers,
        )
        if resp.status_code == 200:
            hits = resp.json().get("hits", {}).get("hits", [])
            if hits:
                out = [f"**SEC EDGAR: {query}** ({len(hits)} results)\n"]
                for i, h in enumerate(hits[:max_results], 1):
                    src = h.get("_source", {})
                    names = src.get("display_names", [])
                    entity = names[0] if names else "?"
                    out.append(
                        f"{i}. **{entity}** - {src.get('form_type', '')}\n"
                        f"   Filed: {src.get('file_date', '')} | File#: {src.get('file_num', '')}"
                    )
                return "\n\n".join(out)
    except Exception:
        pass

    return (
        f"**SEC EDGAR search links for: {query}**\n"
        f"Full-text: https://efts.sec.gov/LATEST/search-index?q={query}&forms={filing_type}\n"
        f"Company: https://www.sec.gov/cgi-bin/browse-edgar?company={query}&type={filing_type or '10-K'}"
    )


@langchain_tool
def search_court_opinions(
    query: str,
    court: str = "",
    max_results: int = 10,
) -> str:
    """Search CourtListener for court opinions. Free.

    Free access to PACER documents: lawsuits, patent disputes, environmental cases.

    Args:
        query: Case name, topic, or party name.
        court: Court filter (e.g. "scotus", "ca9", "nysd"). Empty for all.
        max_results: Maximum results (default 10).

    Returns:
        Court opinions with case names, dates, citations, and links.
    """
    params: dict = {
        "q": query,
        "page_size": min(max_results, 20),
        "order_by": "score desc",
    }
    if court:
        params["court"] = court

    headers: dict = {"User-Agent": _UA}
    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Token {token}"

    try:
        resp = _HTTP.get(
            "https://www.courtlistener.com/api/rest/v4/search/",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"CourtListener error: {exc}"

    results = data.get("results", [])
    if not results:
        return f"No court opinions for: {query}"

    out = [f"**Court opinions: {query}** ({len(results)} results)\n"]
    for i, c in enumerate(results[:max_results], 1):
        name = c.get("caseName", c.get("case_name", "?"))
        date_filed = c.get("dateFiled", c.get("date_filed", ""))
        citation = c.get("citation", [""])
        if isinstance(citation, list):
            citation = citation[0] if citation else ""
        snippet = re.sub(r"<[^>]+>", "", (c.get("snippet", "") or "")[:200]).strip()
        abs_url = c.get("absolute_url", "")
        url = f"https://www.courtlistener.com{abs_url}" if abs_url else ""

        out.append(
            f"{i}. **{name}**\n"
            f"   Court: {c.get('court', '')} | Filed: {date_filed}"
            + (f"\n   Citation: {citation}" if citation else "")
            + (f"\n   URL: {url}" if url else "")
            + (f"\n   {snippet}..." if snippet else "")
        )
    return "\n\n".join(out)


@langchain_tool
def search_offshore_leaks(query: str, max_results: int = 10) -> str:
    """Search ICIJ Offshore Leaks (Panama/Paradise/Pandora Papers). Free.

    810,000+ offshore entities — shell companies, tax havens, hidden wealth.

    Args:
        query: Person name, company name, or country.
        max_results: Maximum results (default 10).

    Returns:
        Matching offshore entities with jurisdictions and data source.
    """
    try:
        resp = _HTTP.get(
            "https://offshoreleaks.icij.org/api/v1/search",
            params={"q": query, "limit": min(max_results, 100)},
            headers={"User-Agent": _UA},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data if isinstance(data, list) else data.get("results", data.get("data", []))
            if results:
                out = [f"**Offshore Leaks: {query}** ({len(results)} results)\n"]
                for i, r in enumerate(results[:max_results], 1):
                    if isinstance(r, dict):
                        out.append(
                            f"{i}. **{r.get('name', r.get('entity', '?'))}**\n"
                            f"   Jurisdiction: {r.get('jurisdiction', '')}\n"
                            f"   Source: {r.get('source', r.get('dataset', ''))}"
                        )
                return "\n\n".join(out)
    except Exception:
        pass

    return (
        f"**ICIJ Offshore Leaks — search: {query}**\n"
        f"Web search: https://offshoreleaks.icij.org/search?q={query}"
    )


# ════════════════════════════════════════════════════════════════════════
# 5. RESEARCH INTEGRITY — free
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def check_retraction(doi: str) -> str:
    """Check if a paper has been retracted. Free.

    Checks OpenAlex and CrossRef for retraction flags. Use before citing.

    Args:
        doi: The DOI to check (e.g. "10.1016/j.cell.2023.01.001").

    Returns:
        Retraction status with details if retracted.
    """
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    # Check OpenAlex
    try:
        resp = _HTTP.get(
            f"https://api.openalex.org/works/doi:{doi}",
            headers={"User-Agent": _UA},
            timeout=15,
        )
        if resp.status_code == 200:
            w = resp.json()
            title = w.get("title", "Unknown")
            if w.get("is_retracted"):
                return (
                    f"RETRACTED: **{title}**\n"
                    f"DOI: {doi}\n"
                    f"Retraction Watch: https://retractionwatch.com/?s={doi}"
                )
            return f"NOT RETRACTED: **{title}** (DOI: {doi})"
    except Exception:
        pass

    # Fallback: CrossRef
    try:
        resp = _HTTP.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": _UA},
            timeout=15,
        )
        if resp.status_code == 200:
            d = resp.json().get("message", {})
            title = " ".join(d.get("title", ["Unknown"]))
            for u in d.get("update-to", []):
                if u.get("type") in ("retraction", "withdrawal"):
                    return f"RETRACTED: **{title}** (DOI: {doi}) — {u.get('type')}"
            rel = d.get("relation", {})
            if "is-retracted-by" in rel:
                return f"RETRACTED: **{title}** (DOI: {doi})"
            return f"NOT RETRACTED: **{title}** (DOI: {doi})"
    except Exception as exc:
        return f"Retraction check failed for {doi}: {exc}"

    return f"Could not verify retraction status for DOI: {doi}"


# ════════════════════════════════════════════════════════════════════════
# 6. ARCHIVES / OSINT — free
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def wayback_search(url: str, limit: int = 5) -> str:
    """Search the Wayback Machine for archived snapshots of a URL. Free.

    Args:
        url: The URL to look up in the Internet Archive.
        limit: Maximum number of snapshots to return (default 5).

    Returns:
        List of archived snapshots with timestamps and archive URLs.
    """
    try:
        resp = _HTTP.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": url,
                "output": "json",
                "limit": min(limit, 20),
                "fl": "timestamp,original,statuscode,mimetype,length",
                "sort": "reverse",
            },
            timeout=30,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        return f"Wayback Machine error: {exc}"

    if not rows or len(rows) <= 1:
        return f"No Wayback Machine snapshots for: {url}"

    header = rows[0]
    out = [f"**Wayback Machine: {url}** ({len(rows) - 1} snapshots)\n"]
    for row in rows[1:]:
        entry = dict(zip(header, row))
        ts = entry.get("timestamp", "")
        formatted_ts = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
        archive_url = f"https://web.archive.org/web/{ts}/{entry.get('original', url)}"
        out.append(
            f"- {formatted_ts} | Status: {entry.get('statuscode', '?')} | "
            f"{archive_url}"
        )
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════
# 7. CONTENT EXTRACTION
# ════════════════════════════════════════════════════════════════════════


@langchain_tool
def jina_read_url(url: str) -> str:
    """Extract clean readable text from any web page using Jina Reader. Free.

    Better than raw fetch for complex pages — strips boilerplate, ads, and
    navigation to return just the article/content text.

    Args:
        url: The URL to read.

    Returns:
        Clean extracted text content from the page.
    """
    try:
        resp = _HTTP.get(
            f"https://r.jina.ai/{url}",
            headers={
                "Accept": "text/plain",
                "User-Agent": _UA,
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.text.strip()
        if len(text) > 15000:
            text = text[:15000] + "\n\n[...truncated...]"
        return text if text else f"No readable content extracted from {url}"
    except Exception as exc:
        return f"Jina Reader error for {url}: {exc}"


# ════════════════════════════════════════════════════════════════════════
# Tool registry — import this list in deepagents_proxy.py
# ════════════════════════════════════════════════════════════════════════

ALL_RESEARCH_TOOLS = [
    # Academic
    openalex_search,
    semantic_scholar_search,
    search_pubmed,
    resolve_doi,
    # Preprints
    search_biorxiv,
    # Government / Legal
    search_clinical_trials,
    search_fda_adverse_events,
    search_sec_filings,
    search_court_opinions,
    search_offshore_leaks,
    # Research integrity
    check_retraction,
    # Archives
    wayback_search,
    # Content extraction
    jina_read_url,
]
