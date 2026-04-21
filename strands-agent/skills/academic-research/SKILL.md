name: academic-research
description: >
  Systematic academic literature search, citation network traversal,
  and research integrity verification. Covers 240M+ works via OpenAlex,
  200M+ papers via Semantic Scholar, PubMed biomedical literature,
  Google Scholar, preprint servers, and document acquisition pipelines.

## When to Activate

- Scientific or molecular research questions
- Literature reviews or meta-analysis requests
- Citation network analysis (who cites whom, seminal papers)
- Pharmacokinetics, pharmacodynamics, mechanism-of-action queries
- Any query mentioning DOIs, PMIDs, journal names, or specific papers
- Cross-referencing claims against published evidence

## Tool Strategy — Academic Research Pipeline

### Phase 1: Literature Discovery (cast the widest net)

Use MULTIPLE databases — each has different coverage:

1. **OpenAlex** (`openalex_search`, `openalex_get_work`) — 240M+ works,
   strongest for citation metadata and open access status. Start here
   for broad literature surveys.

2. **Semantic Scholar** (`ss_search_papers`, `ss_get_paper`) — 200M+ papers,
   strongest for AI/CS/biomedical. Use `ss_get_recommendations` for
   finding related work the keyword search missed.

3. **PubMed** (`search_pubmed`, `pubmed_get_abstract`) — gold standard
   for biomedical and life sciences. Use for anything involving drugs,
   clinical outcomes, molecular biology.

4. **Google Scholar** (`search_google_scholar`) — broadest coverage
   including books, theses, grey literature. Use as a sweep to catch
   what the specialized databases miss.

5. **arXiv** (`arxiv_search_papers`) — preprints in physics, math, CS,
   biology. Use for cutting-edge work not yet in journals.

**Critical:** Do NOT rely on a single database. A paper in PubMed may
not be in Semantic Scholar, and vice versa. The overlap between major
databases is only ~60-70%.

### Phase 2: Citation Network Traversal

Once you have key papers, trace the citation network:

1. **Forward citations** (`ss_get_paper_citations`, `openalex_citation_network`)
   — who cited this paper? Finds newer work building on the finding.

2. **Backward references** (`ss_get_paper_references`) — what did this
   paper cite? Finds foundational work and methodology papers.

3. **Recommendations** (`ss_get_recommendations`, `semantic_scholar_recommend`)
   — algorithmically similar papers that keyword search misses.

**The citation graph reveals what keyword search cannot:** a paper about
"GLP-1 receptor signaling in pancreatic beta cells" won't appear in a
search for "semaglutide weight loss" even though it's directly relevant
to understanding the mechanism.

### Phase 3: Document Acquisition

For papers that need full-text analysis:

1. **Open access first** (`search_open_access`, `download_paper`) — tries
   Unpaywall, CORE, PMC, institutional repositories.

2. **DOI resolution** (`resolve_doi_metadata`) — get publisher metadata,
   check if open access version exists.

3. **Preprint versions** (`search_biorxiv`, `search_chemrxiv`) — many
   papers have free preprint versions even if the journal version is
   paywalled.

4. **Alternative sources** (`annas_archive_search`, `search_zenodo`) —
   broader document search.

### Phase 4: Research Integrity

Before citing any paper, verify its status:

1. **Retraction check** (`check_retraction`) — is the paper retracted?
   ~50,000 papers have been retracted; citing one is a serious error.

2. **Batch checking** (`batch_check_retractions`) — verify multiple DOIs
   at once when you have a reference list.

3. **Retraction search** (`search_retractions`) — find retracted papers
   on a specific topic (e.g., "how many ivermectin papers were retracted?").

### Phase 5: Knowledge Synthesis

- **Store findings** (`store_finding`) — persist each significant finding
  with source DOI and confidence level.
- **Knowledge graph** (`add_entity`, `add_edge`, `query_graph`) — build
  a structured map of concepts, their relationships, and evidence quality.
- **Gap analysis** (`find_gaps`) — identify entities with few connections
  that need more research.

## Common Pitfalls

- **Single-database bias:** Searching only PubMed misses ~30-40% of
  relevant literature. Always use at least 2 databases.
- **Ignoring preprints:** Cutting-edge findings are often 6-18 months
  ahead of journal publication. Check bioRxiv/medRxiv.
- **Not checking retractions:** ~1 in 1000 papers is retracted. For
  controversial topics, the rate is much higher.
- **Keyword myopia:** Citation network traversal finds papers that
  use different terminology for the same concepts.
