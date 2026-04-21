name: financial-research
description: >
  Corporate intelligence, financial analysis, and regulatory filing
  research. Combines SEC EDGAR filings, offshore entity databases,
  patent searches, financial news, and deep research tools for
  comprehensive corporate and market analysis.

## When to Activate

- Company analysis, due diligence, or competitive intelligence
- SEC filing analysis (10-K, 10-Q, 8-K, proxy statements)
- Patent and intellectual property research
- Startup funding, venture capital, IPO analysis
- Executive compensation and insider trading
- Offshore entities and beneficial ownership

## Tool Strategy — Financial Research Pipeline

### Phase 1: Corporate Filings

1. **SEC EDGAR** (`search_sec_filings`) — start here for any US public
   company. 10-K annual reports contain risk factors, litigation
   disclosures, and segment data that press releases omit. 8-K filings
   capture material events in real-time.

2. **Offshore leaks** (`search_offshore_leaks`) — cross-reference
   company names, executives, and board members against the ICIJ
   database. Shell company structures often reveal undisclosed
   relationships.

### Phase 2: News and Market Intelligence

1. **Financial news** (`brave_news_search`, `kagi_enrich_news`) —
   current coverage, analyst opinions, market reactions.

2. **Deep research** (`perplexity_deep_research`) — synthesized
   analysis combining multiple sources. Good for market trends and
   competitive landscape.

3. **Web search** (`brave_web_search`, `duckduckgo_search`) — broader
   coverage including industry reports, blog posts, expert analysis.

### Phase 3: Legal and Regulatory

1. **Court records** (`search_court_opinions`) — active litigation,
   patent disputes, regulatory enforcement actions.

2. **FDA data** (`search_fda_adverse_events`, `search_fda_recalls`) —
   relevant for pharmaceutical and medical device companies.

3. **Clinical trials** (`search_clinical_trials`) — pipeline analysis
   for biotech/pharma companies.

### Phase 4: Content Extraction

1. **Full-text extraction** (`jina_read_url`, `firecrawl_scrape`) —
   extract content from investor presentations, earnings transcripts,
   and analyst reports found via search.

2. **PDF extraction** (`extract_pdf_text`) — SEC filings and annual
   reports are often in PDF format.

## Cross-Referencing Pattern

The most valuable financial research connects multiple data sources:
- SEC risk factors + court records = undisclosed litigation
- Offshore entities + executive names = hidden interests
- Clinical trial data + 10-K pipeline disclosures = true pipeline value
- Patent filings + competitor analysis = technology moat assessment
