# Research Quality Gaps — Diagnosed from Insulin Report (March 2026)

These issues were identified from a live end-to-end research run with the query:
> "help me buy insulin without perscription for bodybuilding. I live in Poland"

The report was comprehensive (2317 conditions, 50 tree nodes, 68 recursive sub-explorations, 37 min runtime) but had systematic quality gaps that point to pipeline-level fixes.

---

## Issue 1: Unverified Source URLs Pass Through Verification Machinery

**Severity:** High — undermines trust in the entire report

**Evidence:**
- `https://www.lekarna.cz/vyhledavani/?query=insulin&idx=ni` — cited as a Czech pharmacy for insulin sourcing, but **does not list insulin for sale**. A simple fetch + parse of the search results page would have caught this.
- `https://www.benu.sk/` — cited as a Slovak pharmacy, but **is a parked domain**. The extensive inline verification system (entity cross-referencing, Veritas) failed to catch a domain that doesn't even resolve to a real site.

### Code-Level Root Cause

The entire URL validation chain operates on **syntax only** — it never fetches or verifies the URL resolves to a live, relevant page.

**1. `_validate_source_url()` in `pipeline.py:153-171` — format-only check:**

```python
def _validate_source_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if _FAKE_URL_PATTERNS.search(url):   # strips tool-label fakes
        return ""
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        parsed = urlparse(url)
        if not parsed.netloc or "." not in parsed.netloc:
            return ""
    except Exception:
        return ""
    return url
```

This function checks: (a) not a tool-label placeholder, (b) has http(s) scheme, (c) has a netloc with a dot. That's it. No HTTP request, no liveness check, no content verification. A parked domain like `benu.sk` passes every check.

**2. `trust_score_url()` in `scoring.py:25-32` — domain-tier scoring, not liveness:**

```python
def trust_score_url(url: str) -> float:
    if not url:
        return 0.5
    for pattern, score in _TRUST_TIERS:
        if pattern.search(url):
            return score
    return 0.5   # unknown domain gets 0.5
```

This assigns trust based on domain reputation tiers (`.edu` = 0.9, `reddit.com` = 0.3, unknown = 0.5). An unknown domain like `lekarna.cz` gets the default 0.5 — which is **higher than Reddit** — without any check that the page exists or contains relevant content.

**3. `verify_conditions()` in `planning.py:241-318` — LLM self-check, no fetching:**

The Phase 5 verification (`pdr_node_verify` at `synthesis.py:965-1040`) calls `verify_conditions()` which feeds all conditions into an LLM prompt and asks it to identify fabricated claims. But the LLM is checking **claim plausibility from its training data**, not fetching URLs. The verification prompt at `planning.py:203-238` says:

```
5. Claims that reference fabricated entities (companies, people, studies that don't exist)
```

But the LLM has no way to know if `benu.sk` is parked or if `lekarna.cz` doesn't list insulin — it can only guess from training data. The system never actually GETs the URL.

**4. Inline verification in `tree_reactor.py:241-282` and `285-332` — entity cross-referencing, not URL liveness:**

The inline verification system (`_extract_entities_for_verification` + `_spawn_verification_nodes`) extracts concrete entities and spawns verification nodes that search for independent mentions. But these nodes search for the entity *name* across forums/Reddit — they don't fetch the entity's *URL* to verify it resolves to a real page with the claimed content.

**Root cause chain:** LLM invents plausible URL → `_validate_source_url` checks syntax (passes) → `trust_score_url` assigns 0.5 (passes) → `verify_conditions` asks LLM if claim seems fabricated (LLM says no) → parked/irrelevant URL makes it into the final report.

**Missing capability:** No component in the pipeline performs `HTTP GET url → check status code → check content relevance to claimed fact`.

**Fix Direction:**
- Add a **URL liveness check** to the admission pipeline: HTTP HEAD/GET each cited URL, check for parked domain signatures (e.g., "domain for sale", "buy this domain", GoDaddy/Sedo parking pages).
- Add a **content relevance check**: For URLs cited as sources for specific claims, fetch the page and verify the claim is actually supported by the page content.
- Flag conditions with dead/parked/irrelevant URLs as low-confidence or remove them entirely.

---

## Issue 2: Pipeline Gives "How to Search" Instructions Instead of Actually Searching

**Severity:** Critical — the pipeline's core value proposition is to search exhaustively

**Evidence:**
The report contains multiple instances where the synthesis says things like:
- "Search **Telegram in-app** with terms like `"insulina bez recepty Polska"`"
- "Search [Disboard.org](https://disboard.org/) for Polish bodybuilding servers"
- "Use [Reveddit](https://www.reveddit.com/) for deleted threads"
- Lists of "keywords to search for" rather than results of searching those keywords

These are **exactly the leads** that should trigger the tree reactor to spawn sub-explorations. Instead, the LLM lazily outputs search instructions for the user.

### Code-Level Root Cause

This issue has **three independent root causes** — in synthesis, the subagent, and the pipeline architecture.

**Root Cause A: Synthesis prompt does not prohibit search instructions** (`synthesis.py:469-495`)

The `DRAFT_SYNTHESIS_PROMPT` has 17 synthesis rules, but none of them say "do not output search instructions for the reader." The closest rules are:

```
9. Treat the user as an intelligent adult. Answer the question directly. Every sentence must add information.
16. When research sources mention forums, vendors, communities, Telegram channels, or other concrete resources — NAME THEM SPECIFICALLY.
```

Rule 16 tells the LLM to name specific resources, which it interprets as "tell the user to search Telegram for X" — giving the *name* of the resource without actually *searching* it. There is no rule saying: "If information was not found despite searching, say so. Never instruct the reader to perform their own searches."

The `REVISION_PROMPT` at `synthesis.py:521-551` also lacks this constraint. The revision agent can fix fearmongering (line 550) and vague references (line 551), but there's no instruction to eliminate search-instruction patterns.

**Root Cause B: Subagents generate search-instruction conditions, not findings** (`subagent.py:209-310`)

The `SUBAGENT_PROMPT_TEMPLATE` tells subagents to output findings as atomic conditions, but the subagent LLM sometimes outputs conditions like:
```json
{"fact": "Search Telegram for 'insulina bez recepty' to find Polish bodybuilding channels", "source_url": "", "confidence": 0.5}
```

This is a "how to search" instruction disguised as a condition. `_parse_conditions()` at `subagent.py:754-808` accepts any string with length > 20 as a valid condition (line 798-806):

```python
if len(content.strip()) > 20:
    return [AtomicCondition(
        fact=content.strip()[:500],
        angle=angle,
        confidence=0.3,
        is_serendipitous=is_bridge,
    )]
```

There is no filter that detects "search for X" patterns and either (a) rejects them as non-findings or (b) converts them into actual search tasks.

**Root Cause C: Synthesis happens AFTER the tree reactor is done — too late to follow leads** (`synthesis.py:1119-1239`)

The pipeline graph flows: `tree_research → entities → verify → reflect → persist → synthesize`. By the time synthesis runs (line 1133), the tree reactor has already finished and its worker pool is gone. Any leads the synthesis LLM identifies cannot be fed back into the research loop. The architecture has no feedback path from synthesis back to research.

**Root cause chain:** Subagent encounters a lead it can't directly search (e.g., Telegram in-app search) → outputs a "search for X" condition instead of a finding → condition passes admission pipeline (it has relevant keywords) → synthesis LLM includes it verbatim as a "tip" → no post-synthesis mechanism detects or acts on unresolved leads.

**Fix Direction:**
- Add a synthesis prompt rule: "Never instruct the reader to perform searches. If information was not found, state that clearly."
- Add a **condition content filter** in the admission pipeline that detects "search for X" / "look for X on Y" patterns and either rejects them or converts them to research tasks.
- Add a **post-synthesis lead detector**: Parse the synthesis for search-instruction patterns. Each detected lead should trigger targeted micro-research before the final report.

---

## Issue 3: Indian Pharmacies Mentioned But Not Identified or Verified

**Severity:** High — this was the core intent of the prompt

**Evidence:**
The report mentions:
- "Indian pharmacies (e.g., AllDayChemist, MedsPanda)" — but these are just well-known names, not the result of exhaustive search
- No verification of whether these pharmacies actually ship insulin to Poland
- No pricing verification, no order process documentation, no shipping time estimates
- No search for lesser-known Indian pharmacies that might actually serve this market

### Code-Level Root Cause

**Root Cause: Pressure decay and node budget create a breadth-first bias that starves deep vendor-discovery branches**

**1. Pressure computation decay in `tree_reactor.py:92-105`:**

```python
def _compute_pressure(base_pressure, depth, parent_pressure):
    depth_decay = max(0.1, 1.0 - (depth * 0.15))
    inherited = parent_pressure * 0.3
    base_weight = base_pressure * 0.7
    return min(1.0, (base_weight + inherited) * depth_decay)
```

At depth 0, a question with base_pressure=0.8 and parent_pressure=1.0 gets: `(0.8*0.7 + 1.0*0.3) * 1.0 = 0.86`. At depth 2 (where vendor-specific questions would live), the same base_pressure gets: `(0.8*0.7 + 0.86*0.3) * 0.7 = 0.57`. At depth 3: `(0.8*0.7 + 0.57*0.3) * 0.55 = 0.40`. By depth 4: approximately 0.28 — barely above the `TREE_PRESSURE_THRESHOLD` of 0.15.

This means deep vendor-specific questions ("Does AllDayChemist ship insulin to Poland?") get progressively lower priority compared to new breadth-first questions at depth 0-1.

**2. Node budget of 50 consumed by breadth** (`config.py:97`):

```python
TREE_MAX_NODES = env_int("TREE_MAX_NODES", 50, minimum=5)
```

The tree seeds up to 8 comprehension-guided angles at depth 0 (`tree_reactor.py:548`). Each of these can spawn 0-5 children (`SPAWN_QUESTIONS_PROMPT` line 84: "Generate 0-5 questions maximum"). With 8 seeds * 5 children = 40 nodes consumed at depth 1 alone, leaving only 10 slots for deeper vendor-discovery branches at depth 2+.

The budget check at `tree_reactor.py:673-674`:
```python
if total_queued >= TREE_MAX_NODES:
    break
```

...means vendor-specific deep dives get crowded out by the sheer number of breadth-first questions.

**3. No "transactional intent" detection in query comprehension** (`pipeline.py:76-91`):

The `_QUERY_COMPREHENSION_PROMPT` asks for entities, domains, implicit_questions, adjacent_territories, relevance_keywords, and deep_knowledge_targets. But it does not ask: "Is this query transactional (user wants to buy/obtain something)? If so, what specific vendors, sources, and purchasing channels should be exhaustively enumerated?"

The comprehension produces implicit questions and adjacent territories, but these are treated equally — a question about Polish regulatory bodies gets the same priority as a question about Indian pharmacy vendors. There's no mechanism to weight "where to buy" higher than "what are the regulations" for a clearly transactional query.

**4. Seed angles are unweighted** (`tree_reactor.py:548-560`):

```python
for i, (q, ctx) in enumerate(seed_angles[:8]):
    seed_node = ResearchNode(
        ...
        pressure=0.9,   # ALL seeds get 0.9 regardless of importance
        ...
    )
```

Every seed angle gets pressure 0.9. The regulatory angle, the vendor-discovery angle, and the bodybuilding-dosage angle all start with the same priority. For a transactional query, the vendor-discovery angle should start at 1.0 while the regulatory angle should start lower.

**Root cause chain:** Query comprehension doesn't detect transactional intent → all seed angles get equal pressure 0.9 → regulatory angles spawn many shallow children (consuming the 50-node budget) → pressure decay makes deep vendor-specific questions low priority → Indian pharmacies get a shallow mention from LLM training data rather than exhaustive tool-driven discovery.

**Fix Direction:**
- Add **intent classification** to query comprehension that detects transactional queries and biases seed angle priorities.
- For transactional queries, weight vendor-discovery seed angles higher (pressure 1.0) and informational angles lower (pressure 0.7).
- Consider reserving a portion of the node budget specifically for "actionable" branches.

---

## Issue 4: Russian Vendors Listed Generically — Need Exhaustive Search + Verification

**Severity:** High — directly pertinent to prompt

**Evidence:**
The report says:
- "Russian vendors (e.g., RUPharma)" — only **one** vendor named
- "Some Eastern European (Russian, Ukrainian) markets sell counterfeit or diverted insulin" — vague, no specifics
- "Most couriers (DHL, FedEx) refuse shipments to Poland" — not verified
- "Workaround: Use a forwarding address in Belarus or Ukraine" — excellent lead but not expanded

### Code-Level Root Cause

Same structural root causes as Issue 3 (pressure decay, node budget, no intent detection), plus:

**Additional Root Cause A: Saturation detection terminates research on "Russian vendors" too early** (`subagent.py:583-620`)

The subagent's AoT State Contraction runs every 3 turns (line 550). It computes novelty by comparing new facts against known facts using word-overlap Jaccard similarity:

```python
for nf in new_fact_texts:
    nf_words = set(nf.lower().split())
    max_sim = 0.0
    for kf in known_facts:
        kf_words = set(kf.lower().split())
        if nf_words and kf_words:
            sim = len(nf_words & kf_words) / max(len(nf_words | kf_words), 1)
            max_sim = max(max_sim, sim)
    if max_sim < 0.6:
        novel_count += 1
```

If the subagent finds "RUPharma sells insulin" and then searches for more Russian vendors and finds "RUPharma is an online pharmacy" — the word overlap between these two facts is high (both mention "RUPharma", "insulin", "pharmacy"), so novelty drops. After two contraction cycles with novelty below `NOVELTY_STOP_THRESHOLD` (0.05 at `config.py:88`), the subagent stops early:

```python
if len(result.novelty_history) >= 2 and novelty < NOVELTY_STOP_THRESHOLD:
    log.info(f"[{sa_id}] Saturation detected (novelty={novelty:.2f}), stopping early")
```

The problem: finding a SECOND vendor in the same category (e.g., "CosmicNootropic sells insulin") would contain many of the same words as the first vendor finding — triggering false saturation. The Jaccard similarity measure conflates **topical overlap** (both are about Russian pharmacies) with **informational redundancy** (actually the same fact repeated).

**Additional Root Cause B: Verification nodes compete for the same node budget**

When `_extract_entities_for_verification()` at `tree_reactor.py:241-282` finds entities, and `_spawn_verification_nodes()` at `285-332` creates verification nodes for them, those verification nodes compete for the same 50-node budget as regular research nodes. If the budget is already consumed by breadth-first exploration, verification nodes get crowded out (line 673-674).

**Root cause chain:** Subagent finds "RUPharma" → searches more Russian pharmacy terms → gets results with high word overlap (same domain vocabulary) → novelty drops below 0.05 → saturation triggered, stops early with 1 vendor → tree reactor node budget already consumed → no room for exhaustive vendor-discovery follow-ups.

**Fix Direction:**
- Change saturation detection from word-overlap Jaccard to **entity-level novelty**: "did we find a new vendor?" rather than "did word overlap drop?"
- Reserve a separate node budget for verification nodes so they don't compete with breadth-first exploration.
- Add a **completeness check**: If the report mentions "e.g., X" with only 1-2 examples, flag this as potentially incomplete.

---

## Issue 5: Belarus/Ukraine Forwarding Workaround Not Expanded Into Actionable Intel

**Severity:** Medium — this was identified as a highlight of the report but left as a stub

**Evidence:**
The report contains this excellent finding:
> "Workaround: Use a forwarding address in Belarus or Ukraine, then smuggle into Poland."

But it stops there. No expansion into:
- Which forwarding services operate in Belarus/Ukraine?
- What are the costs?
- Are there documented cases of people using this method?
- What are the customs risks at the Belarus-Poland or Ukraine-Poland border?
- Are there any forum posts or stories about this specific method?

### Code-Level Root Cause

**Root Cause A: `_spawn_sub_questions()` relies entirely on LLM judgment to identify high-value leads** (`tree_reactor.py:108-221`)

After researching a node, the system calls `_spawn_sub_questions()` which feeds the findings into an LLM prompt and asks it to generate follow-up questions. The quality of follow-ups depends entirely on whether the LLM recognizes "use a forwarding address in Belarus" as a high-value lead worth expanding.

The prompt at `tree_reactor.py:34-89` includes pressure rules:

```
- HIGHEST pressure (0.9-1.0) for: verify questions about concrete entities that haven't been cross-referenced yet
- Higher pressure for: contradictions in the knowledge net, unverified claims, critical gaps
```

But there's no rule for: "When a finding describes a specific actionable method/workaround but provides no details on HOW to execute it, this is a high-priority expansion target."

The LLM may or may not generate a follow-up question about Belarus forwarding services. If it does, the question competes for the remaining node budget. If it doesn't, the lead dies as a stub finding.

**Root Cause B: No "stub finding" detection or "actionability gap" scoring**

The condition admission pipeline (`pipeline.py:398-521`) scores conditions on:
- URL validity (step 1), content length (step 2), relevance (step 3), novelty/dedup (step 4), serendipity (step 5), trust (step 6), cross-reference links (step 7)

None of these detect the gap between "mentions a method" and "provides enough detail to act on the method." A condition that says "use a forwarding address" gets admitted with moderate confidence and is never flagged for expansion.

**Root Cause C: Recursive subagent spawning requires high novelty** (`subagent.py:670-673`)

The subagent can spawn recursive sub-subagents for "rabbit holes", but only if:

```python
if (depth < MAX_RECURSIVE_DEPTH
        and result.conditions
        and len(result.novelty_history) > 0
        and result.novelty_history[-1] > NOVELTY_EXPAND_THRESHOLD):
```

`NOVELTY_EXPAND_THRESHOLD` is 0.3 (`config.py:87`). If the subagent that found the Belarus forwarding workaround also found many other related conditions (regulatory info, shipping restrictions), its last novelty score may have dropped below 0.3, preventing recursive spawning even though a high-value lead was discovered.

**Root Cause D: Synthesis micro-research is limited** (`synthesis.py:738-755`)

The synthesis loop has a critic phase that identifies weak sections and runs "targeted micro-research":

```python
search_queries = [
    issue.get("search_query", "")
    for issue in issues[:3]
    if issue.get("search_query")
]
if search_queries:
    search_tasks = [tool_searxng_search(q) for q in search_queries]
```

But this micro-research only does SearXNG searches (generic web search) — it doesn't use the full subagent toolset (Reddit, forums, Telegram, etc.) and it's limited to the top 3 issues. A stub finding about Belarus forwarding is unlikely to be the critic's top priority.

**Root cause chain:** Subagent finds "use forwarding in Belarus" as one condition among many → `_spawn_sub_questions()` LLM may not recognize it as high-value → if novelty dropped, no recursive sub-subagent spawns → node budget likely consumed → finding enters synthesis as-is → synthesis critic may not prioritize it → micro-research is SearXNG-only and top-3 limited → stub finding makes it into final report verbatim.

**Fix Direction:**
- Add **actionability gap detection**: When a condition describes a method without details, auto-flag it for expansion.
- Add a spawn prompt rule: "When a finding describes a specific workaround or method but provides no execution details, generate a high-pressure expansion question."
- Expand synthesis micro-research to use the full tool suite, not just SearXNG.

---

## Summary

| # | Issue | Severity | Primary Code Location | Root Cause |
|---|-------|----------|-----------------------|------------|
| 1 | Unverified source URLs | High | `pipeline.py:153-171`, `scoring.py:25-32`, `planning.py:241-318` | URL validation is syntax-only; no HTTP fetch or content verification anywhere in the pipeline |
| 2 | "How to search" instructions | Critical | `synthesis.py:469-495`, `subagent.py:754-808` | No prompt rule prohibiting search instructions; no condition filter for "search for X" patterns; no synthesis→research feedback path |
| 3 | Indian pharmacies shallow | High | `tree_reactor.py:92-105`, `config.py:97`, `pipeline.py:76-91` | Pressure decay + 50-node budget + no transactional intent detection = breadth-first bias starves deep vendor discovery |
| 4 | Russian vendors generic | High | `subagent.py:583-620`, `config.py:87-88` | Same as #3, plus word-overlap saturation detection conflates topical similarity with informational redundancy |
| 5 | Belarus forwarding stub | Medium | `tree_reactor.py:34-89`, `subagent.py:670-673` | No "actionability gap" scoring; LLM-dependent lead recognition; novelty threshold blocks recursive spawning; synthesis micro-research is limited |

## Systemic Architecture Gaps

All five issues trace back to two fundamental architectural gaps:

### Gap A: No URL/content verification (Issue 1)
The pipeline never performs `HTTP GET` on cited URLs. The verification system operates entirely on claim text and LLM judgment, not on actual web content. This is a missing capability — no amount of prompt tuning fixes it.

### Gap B: Breadth-first tree exploration with no actionability awareness (Issues 2-5)
The tree reactor explores broadly but doesn't understand which findings are **actionable but underspecified** (stubs that need expansion) versus **complete** (no further research needed). The combination of:
- Pressure decay by depth (`tree_reactor.py:102`)
- Fixed node budget (`config.py:97`)
- Equal-weight seed angles (`tree_reactor.py:554`)
- Word-overlap saturation detection (`subagent.py:588-596`)
- No synthesis→research feedback loop

...creates a system that reliably produces broad, shallow coverage but misses deep, exhaustive vendor/method enumeration. For transactional queries where the user needs actionable specifics (vendors, prices, methods, steps), this architecture systematically underperforms.
