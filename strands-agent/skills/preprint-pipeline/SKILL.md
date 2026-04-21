name: preprint-pipeline
description: >
  Systematic search across preprint servers (bioRxiv, medRxiv, ChemRxiv,
  SSRN, OSF Preprints) with cross-referencing against published versions
  and retraction databases. Preprints are 6-18 months ahead of journal
  publication — this skill ensures cutting-edge research is captured.

## When to Activate

- Queries about very recent research (last 6-18 months)
- Topics where preprints are primary sources (COVID, AI, genomics)
- Requests for "latest" or "newest" findings
- Cross-referencing preprint claims against peer-reviewed versions
- Checking if a preprint was ever published or retracted

## Tool Strategy — Preprint Pipeline

### Phase 1: Multi-Server Search

Each preprint server covers different disciplines:

1. **bioRxiv/medRxiv** (`search_biorxiv`) — biology and medicine.
   Use `server="biorxiv"` for basic biology, `server="medrxiv"` for
   clinical/medical. These are the highest-volume preprint servers.

2. **ChemRxiv** (`search_chemrxiv`) — chemistry, materials science,
   chemical engineering. Smaller but highly specialized.

3. **SSRN** (`search_ssrn`) — social sciences, economics, law,
   accounting. The dominant preprint server for non-STEM fields.

4. **OSF Preprints** (`search_osf_preprints`) — multi-discipline
   aggregator. Use `list_osf_providers` to see all 30+ preprint
   services indexed.

5. **arXiv** (`arxiv_search_papers`) — physics, mathematics, CS,
   quantitative biology, statistics. The original preprint server.

**Critical:** Search at least 2-3 servers. A preprint about drug
metabolism might be on bioRxiv (biology angle), ChemRxiv (chemistry
angle), or medRxiv (clinical angle).

### Phase 2: Cross-Reference with Published Literature

Preprints may have been:
- **Published** — check PubMed (`search_pubmed`) and OpenAlex
  (`openalex_search`) for the journal version, which may have
  significant revisions.
- **Retracted** — check `check_retraction` with the DOI.
- **Updated** — bioRxiv/medRxiv preprints can have multiple versions.
  Use `search_biorxiv_by_doi` to get version history.

### Phase 3: Quality Assessment

Preprints are NOT peer-reviewed. Flag the following:
- Sample sizes (small n = preliminary)
- Statistical methods (p-hacking, multiple comparisons)
- Conflicts of interest (industry-funded preprints)
- Whether the preprint was ever published (unpublished after 2+ years
  is a red flag)

Always note "preprint — not yet peer-reviewed" when citing.

## Common Pitfalls

- **Treating preprints as established science:** Always flag preprint
  status explicitly. Many preprints are never published.
- **Missing the published version:** A preprint from 2023 may have a
  peer-reviewed version with corrections. Always cross-check.
- **Server bias:** Searching only bioRxiv misses SSRN, ChemRxiv, and
  OSF content. Cast a wide net.
