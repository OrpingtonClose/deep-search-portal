# RFC: Migrate Research Pipeline to LangGraph StateGraph

**Status:** Proposed
**Priority:** Enhancement
**Created:** 2026-03-21

## Summary

Refactor the research pipeline from the current custom async orchestration into a **maximally abstracted LangGraph StateGraph**. This enables:

1. **Visual execution graph** — LangSmith (or compatible tools) can render the DAG of node execution in real-time, showing which phases are active, completed, or failed
2. **Maximal abstraction** — each pipeline phase (decomposition, tree research, entity extraction, verification, synthesis, persistence) becomes a discrete graph node with typed state transitions
3. **Conditional routing** — LangGraph conditional edges replace the current if/else orchestration logic
4. **Checkpoint/resume** — built-in state persistence allows resuming interrupted research runs
5. **Streaming** — LangGraph native streaming replaces the current manual SSE implementation

## Current Architecture

The pipeline is orchestrated by custom Python code in `proxies/tools/pipeline.py` and `proxies/persistent_deep_research_proxy.py`:

```
user query -> decomposition -> tree_research (recursive subagents) -> entity_extraction -> verification -> synthesis -> persistence
```

Each phase is a Python async function called sequentially. State is passed via dictionaries and the `ResearchCollector` class.

## Proposed Architecture

```python
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    query: str
    decomposed_questions: list[str]
    findings: list[Finding]
    entities: list[Entity]
    conditions: list[AtomicCondition]
    verified_conditions: list[AtomicCondition]
    report: str
    neo4j_stored: int
    langfuse_trace_id: str

graph = StateGraph(ResearchState)
graph.add_node("decompose", decompose_query)
graph.add_node("tree_research", run_tree_research)
graph.add_node("extract_entities", extract_entities)
graph.add_node("verify", verify_conditions)
graph.add_node("synthesize", synthesize_report)
graph.add_node("persist", persist_to_neo4j)

graph.add_edge("decompose", "tree_research")
graph.add_edge("tree_research", "extract_entities")
graph.add_edge("extract_entities", "verify")
graph.add_conditional_edges("verify", should_re_research_or_synthesize)
graph.add_edge("synthesize", "persist")
graph.add_edge("persist", END)

graph.set_entry_point("decompose")
app = graph.compile()
```

## Key Design Decisions

- **Maximal abstraction**: Each node should be a pure function `(ResearchState) -> ResearchState` with no side effects beyond the state update. IO (Neo4j, HTTP, LLM calls) happens inside nodes but state transitions are explicit.
- **Subgraph for tree research**: The recursive tree research phase should itself be a subgraph with conditional depth-based routing
- **Streaming via LangGraph**: Replace manual SSE chunk emission with LangGraph's `stream_events` or `astream` API
- **Langfuse integration**: LangGraph's LangChain callback integration means Langfuse traces will automatically capture the graph execution structure
- **LangSmith compatibility**: With a proper StateGraph, LangSmith can render the visual execution graph that shows real-time node activation

## Acceptance Criteria

- [ ] All pipeline phases are LangGraph nodes with typed `ResearchState`
- [ ] Tree research is a subgraph with conditional depth routing
- [ ] SSE streaming works via LangGraph's native streaming API
- [ ] Langfuse traces show the graph structure (not just flat spans)
- [ ] All existing tests pass with the new architecture
- [ ] LangSmith can render the execution graph (if configured)
- [ ] No regression in research quality or performance

## References

- [LangGraph docs](https://python.langchain.com/docs/langgraph)
- [LangGraph StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- Current pipeline: `proxies/tools/pipeline.py`
- Current proxy: `proxies/persistent_deep_research_proxy.py`
- Current tree reactor: `proxies/tools/tree_reactor.py`
