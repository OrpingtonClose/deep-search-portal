name: government-data
description: >
  Structured queries against government databases — ClinicalTrials.gov,
  OpenFDA (FAERS adverse events, recalls), CourtListener (federal court
  opinions), SEC EDGAR (corporate filings), and ICIJ Offshore Leaks.
  These databases contain ground-truth regulatory data that web search
  cannot surface.

## When to Activate

- Clinical trial outcomes or trial design questions
- Drug safety signals, adverse events, FDA actions
- Legal proceedings, court rulings, case law
- Corporate filings, SEC disclosures, insider trading
- Offshore entities, beneficial ownership, shell companies
- Regulatory compliance or enforcement actions

## Tool Strategy — Government Data Pipeline

### ClinicalTrials.gov

1. **Search trials** (`search_clinical_trials`) — find trials by
   condition, intervention, sponsor, or NCT ID. Returns structured
   data: phase, status, enrollment, dates, sponsor.

2. **Get results** (`get_trial_results`) — retrieve outcome measures,
   adverse events, and statistical results for completed trials.
   This is the most underused tool — it contains data that published
   papers often don't include (e.g., all adverse events, not just
   the ones the authors chose to highlight).

**When to use:** Any query about drug efficacy, safety profiles,
ongoing research, or pharmaceutical company pipelines. The trial
registry is often 1-2 years ahead of published results.

### OpenFDA (FAERS)

1. **Adverse events** (`search_fda_adverse_events`) — search the FDA
   Adverse Event Reporting System. Contains millions of reports of
   drug side effects from real-world use (not just clinical trials).

2. **Recalls** (`search_fda_recalls`) — product recalls and safety
   alerts. Covers drugs, devices, food, cosmetics.

**When to use:** Drug safety questions, side effect profiles,
drug interactions. FAERS data is noisy (voluntary reporting) but
captures signals that clinical trials miss due to small sample sizes
or short durations.

### CourtListener / RECAP

1. **Court opinions** (`search_court_opinions`) — search federal court
   opinions and orders. Covers all federal circuits.

**When to use:** Legal questions, regulatory enforcement, patent
litigation, corporate disputes. Court opinions contain detailed
factual findings that news articles summarize poorly.

### SEC EDGAR

1. **SEC filings** (`search_sec_filings`) — search 10-K, 10-Q, 8-K,
   proxy statements, and other corporate disclosures.

**When to use:** Corporate intelligence, financial analysis, executive
compensation, risk factors, material events. 10-K risk factor sections
often disclose information companies don't publicize.

### ICIJ Offshore Leaks

1. **Offshore search** (`search_offshore_leaks`) — search the ICIJ
   database of offshore entities from Panama Papers, Paradise Papers,
   Pandora Papers, and other leaks.

**When to use:** Beneficial ownership, shell company structures,
tax haven entities, politically exposed persons.

## Cross-Referencing Strategy

Government databases are most powerful when cross-referenced:

- **Drug safety:** ClinicalTrials.gov (trial data) + FAERS (real-world)
  + PubMed (published analysis) = complete safety picture
- **Corporate investigation:** SEC filings (disclosures) + court records
  (litigation) + offshore leaks (hidden entities) = full corporate picture
- **Regulatory action:** FDA recalls + court opinions + SEC filings =
  enforcement timeline
