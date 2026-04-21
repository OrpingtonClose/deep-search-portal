---
name: forum-mining
description: >
  Dedicated forum scraping for practitioner knowledge — protocols, bloodwork,
  dosing adjustments, side effect management, vendor reviews, and real-world
  experience reports. Covers 14 bodybuilding/PED forums across 6 languages
  (EN, DE, PL, ES, FR, RU). Use when the query involves substances, protocols,
  or practices where practitioner experience is more valuable than academic
  literature — PEDs, hormone therapy, training under gear, insulin/GH timing,
  supplement stacking, vendor sourcing.
allowed-tools: >
  forum_search forum_read_thread forum_deep_dive forum_list
  duckduckgo_search jina_read_url reddit_search
  reddit_get_subreddit_posts reddit_get_post_details
  store_finding read_findings add_entity add_edge query_graph find_gaps
---

# Forum Mining — Practitioner Knowledge Extraction

Use this skill when researching topics where **real-world experience reports**
are the primary source of truth. Academic literature is secondary — the people
who actually run cycles, adjust doses based on bloodwork, and manage side
effects are the experts.

## When to Activate

- PED protocols (steroids, SARMs, peptides, GH, insulin)
- Hormone therapy and TRT optimization
- Training programming under gear (exercise selection, volume, recovery)
- Nutrient timing with pharmacological agents
- Vendor sourcing and product quality reviews
- Side effect management and harm reduction
- Any query where "what do practitioners actually do" matters more than
  "what does the literature say"

## Forum Registry

### English (8 forums)

| Forum | Domain | Speciality |
|-------|--------|-----------|
| MesoRx | meso-rx.org | Gold standard for harm reduction, protocols, bloodwork |
| EliteFitness | elitefitness.com | Large community, vendor reviews |
| Professional Muscle | professionalmuscle.com | Advanced/competitive users |
| AnabolicMinds | anabolicminds.com | Supplements + PED discussion |
| T-Nation | forums.t-nation.com | Training + pharma subforum |
| ThinkSteroids | thinksteroids.com | Evidence-based PED discussion |
| UK-Muscle | uk-muscle.co.uk | UK community, NHS TRT discussion |
| Evolutionary | evolutionary.org | Protocols + stacking guides |

### International (6 forums)

| Forum | Domain | Language | Speciality |
|-------|--------|----------|-----------|
| Extrem-Bodybuilding | extrem-bodybuilding.de | DE | Team-Andro successor, largest German forum |
| SFD | sfd.pl | PL | Largest Polish fitness forum |
| Hipertrofia | hipertrofia.org | ES | Spanish bodybuilding |
| Musculacion | musculacion.net | ES | Spanish training + PED |
| Superphysique | superphysique.org | FR | French bodybuilding |
| IronPharm | ironpharm.org | RU | Russian PED community |

## Protocol

### Phase 1 — Broad sweep with `forum_search`

Start with `forum_search(query="...", forums="all")` to see which forums
have relevant discussion. Use natural language queries — DuckDuckGo handles
site-scoping.

**Search in multiple angles:**
- The compound/protocol name directly: `"trenbolone acetate cycle"`
- Common abbreviations: `"tren ace blast"`
- Bloodwork context: `"tren bloodwork liver values"`
- Side effect management: `"tren sides insomnia management"`
- Stacking context: `"tren + GH + insulin timing"`
- Dosing specifics: `"tren dosage first cycle mg"`

**For international forums, search in the local language:**
- German: `"Trenbolon Kur Erfahrung"` (cycle experience)
- Polish: `"trenbolon cykl dawkowanie"` (cycle dosing)
- Spanish: `"trembolona ciclo dosis"` (cycle dose)
- French: `"trenbolone cycle dosage avis"` (cycle dosage opinion)
- Russian: `"тренболон курс дозировка"` (course dosing)

### Phase 2 — Deep extraction with `forum_deep_dive`

For the most promising topics, use `forum_deep_dive` to search AND extract
full thread text in one call. Target 3-5 threads per sub-topic.

Prioritize threads with:
- Detailed bloodwork numbers (pre/mid/post cycle)
- Multi-week logs with dosing adjustments
- Experienced users (5+ years, multiple cycles)
- Threads with debate/disagreement (signal of nuance)
- Threads that reference specific protocols by name

### Phase 3 — Targeted thread extraction with `forum_read_thread`

For specific high-value threads found during search, extract the full text.
Forum threads can be 50+ posts — the full context matters for understanding
protocol evolution and community consensus.

### Phase 4 — Cross-reference across forums

The same protocol discussed on MesoRx vs EliteFitness vs a Russian forum
will have different perspectives. Cross-reference to find:
- Points of consensus (these are likely reliable)
- Points of disagreement (these need deeper investigation)
- Regional variations (different access to compounds, different medical
  oversight, different training traditions)

### Phase 5 — Store findings

Use `store_finding` for every substantive protocol detail found. Include:
- Exact dosages, frequencies, durations
- Bloodwork values (reference ranges AND actual reported values)
- Side effect incidence and management approaches
- Source forum and thread URL for provenance

Build the knowledge graph with `add_entity` and `add_edge` to connect:
- Compounds → protocols → side effects → management strategies
- Users → experience level → reported outcomes
- Forums → consensus positions → supporting evidence

## Quality Signals

**High-value posts:**
- Include specific numbers (mg, IU, ml, blood values)
- Reference personal bloodwork or experience logs
- Acknowledge trade-offs and risks
- Cite other users' experiences for cross-validation

**Low-value posts (skip or deprioritize):**
- Generic advice without personal experience
- Single-sentence responses
- Posts that just link to articles without commentary
- Vendor promotion without substantive content

## Censorship Notes

- `forum_search` uses DuckDuckGo site-scoped search — **fully uncensored**
- `forum_read_thread` uses Jina Reader — **no content filtering**
- These tools will NOT reject health/PED queries (unlike Exa or Google)
- International forums are especially valuable because they often have
  less moderation of PED discussion than English-language forums
