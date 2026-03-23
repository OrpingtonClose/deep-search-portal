"""
tools/ package — decomposed research logic from persistent_deep_research_proxy.py.

Module map:
    config          — env vars, constants, LLM factories, shared state
    models          — CrossRef, AtomicCondition, SubagentResult, ResearchNode
    scoring         — trust_score_url, serendipity_score
    persistence     — JSONL logging, Neo4j storage/retrieval, document ingestion
    tool_defs       — NATIVE_TOOLS, LANGCHAIN_TOOLS definitions
    moderation      — content classifier, moderate_query, commercial SERP
    search_tools    — searxng_query, search formatting, news_search, fetch_webpage
    web_fetch       — enhanced_web_fetch, multi-tier fallback chain, 4chan archives
    search_tools2   — Twitter/X, arXiv, Wayback, Wikidata, HN, SE, PubMed, etc.
    tool_executor   — retry wrappers, PDF extraction, execute_tool dispatcher
    llm             — call_llm, message conversion
    pipeline        — condition admission, ConditionStore, QueryComprehension
    planning        — question router, entity extraction, citation verification
    subagent        — planning agent, AoT reflection, run_subagent
    tree_reactor    — tree research reactor
    ruflo_synthesis — ruflo gossip protocol for chunked synthesis of large finding sets
    synthesis       — live findings, heartbeat, draft-synthesis-revision, LangGraph pipeline
    conversation    — conversation state persistence, follow-up detection, snapshot store
"""
