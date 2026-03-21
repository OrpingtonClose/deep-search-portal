"""Tests for the tree-based research reactor, pressure scoring,
enhanced web scraping abstraction, and curated thought output.

All external dependencies (LLM, Neo4j, SearXNG, Apify, Bright Data)
are mocked — no services need to be running.
"""

import asyncio
import sys
import os
import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Add proxies to path so we can import persistent_deep_research_proxy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxies"))

# Mock shared module before importing the proxy (it calls require_env at import time)
_mock_shared = MagicMock()
_mock_shared.setup_logging.return_value = MagicMock()
_mock_shared.require_env.return_value = "test-key"
_mock_shared.env_int.side_effect = lambda name, default, **kw: default
_mock_shared.http_client.return_value = AsyncMock()
_mock_shared.create_app.return_value = MagicMock()
_mock_shared.register_standard_routes = MagicMock()
_mock_shared.make_sse_chunk = MagicMock(side_effect=lambda content, **kw: f"data: {content}\n\n")
_mock_shared.ConcurrencyLimiter = MagicMock
_mock_shared.RequestTracker = MagicMock
_mock_shared.is_utility_request = MagicMock(return_value=False)
_mock_shared.stream_passthrough = MagicMock()
sys.modules["shared"] = _mock_shared

# Mock knowledge_client before import
_mock_kc = MagicMock()
_mock_kc.search_research_conditions = AsyncMock(return_value=[])
_mock_kc.store_research_conditions = AsyncMock(return_value=0)
_mock_kc.get_graph_neighbors = AsyncMock(return_value=[])
_mock_kc.store_entities = AsyncMock(return_value=(0, 0))
sys.modules["knowledge_client"] = _mock_kc

import persistent_deep_research_proxy as pdr


# ============================================================================
# Test: ResearchNode
# ============================================================================

class TestResearchNode:
    def test_creation(self):
        node = pdr.ResearchNode(
            id="n1",
            question="What is X?",
            context="Root question",
            depth=0,
            pressure=0.8,
        )
        assert node.id == "n1"
        assert node.question == "What is X?"
        assert node.depth == 0
        assert node.pressure == 0.8
        assert node.status == "pending"
        assert node.parent_id is None

    def test_ordering_higher_pressure_first(self):
        """ResearchNode.__lt__ should sort higher pressure before lower."""
        high = pdr.ResearchNode(id="h", question="q", context="c", depth=0, pressure=0.9)
        low = pdr.ResearchNode(id="l", question="q", context="c", depth=0, pressure=0.3)
        assert high < low  # higher pressure = higher priority = "less than"
        assert not (low < high)

    def test_equal_pressure(self):
        a = pdr.ResearchNode(id="a", question="q", context="c", depth=0, pressure=0.5)
        b = pdr.ResearchNode(id="b", question="q", context="c", depth=0, pressure=0.5)
        assert not (a < b)
        assert not (b < a)

    def test_priority_queue_ordering(self):
        """Nodes should come out of a PriorityQueue highest-pressure first."""
        q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        nodes = [
            pdr.ResearchNode(id="low", question="q", context="c", depth=0, pressure=0.2),
            pdr.ResearchNode(id="high", question="q", context="c", depth=0, pressure=0.9),
            pdr.ResearchNode(id="mid", question="q", context="c", depth=0, pressure=0.5),
        ]
        for n in nodes:
            q.put_nowait(n)

        order = [q.get_nowait().id for _ in range(3)]
        assert order == ["high", "mid", "low"]

    def test_child_node_with_parent(self):
        parent = pdr.ResearchNode(id="p1", question="q", context="c", depth=0, pressure=1.0)
        child = pdr.ResearchNode(
            id="c1", question="sub-q", context="because",
            depth=1, pressure=0.6, parent_id=parent.id,
        )
        assert child.parent_id == "p1"
        assert child.depth == 1


# ============================================================================
# Test: Pressure scoring
# ============================================================================

class TestComputePressure:
    def test_root_node_full_pressure(self):
        """Depth 0 should have no decay."""
        result = pdr._compute_pressure(base_pressure=1.0, depth=0, parent_pressure=1.0)
        # depth_decay = max(0.1, 1.0 - 0*0.15) = 1.0
        # base_weight = 1.0 * 0.7 = 0.7
        # inherited = 1.0 * 0.3 = 0.3
        # result = min(1.0, (0.7 + 0.3) * 1.0) = 1.0
        assert result == pytest.approx(1.0)

    def test_depth_decay(self):
        """Deeper nodes should get lower pressure."""
        shallow = pdr._compute_pressure(0.8, depth=1, parent_pressure=0.8)
        deep = pdr._compute_pressure(0.8, depth=4, parent_pressure=0.8)
        assert shallow > deep

    def test_zero_base_pressure(self):
        result = pdr._compute_pressure(0.0, depth=0, parent_pressure=0.0)
        assert result == pytest.approx(0.0)

    def test_clamped_to_one(self):
        """Result should never exceed 1.0."""
        result = pdr._compute_pressure(1.0, depth=0, parent_pressure=1.0)
        assert result <= 1.0

    def test_minimum_depth_decay(self):
        """Even at extreme depth, decay floor is 0.1."""
        result = pdr._compute_pressure(1.0, depth=100, parent_pressure=1.0)
        # depth_decay = max(0.1, 1.0 - 100*0.15) = 0.1
        assert result > 0.0
        assert result == pytest.approx(min(1.0, (0.7 + 0.3) * 0.1))

    def test_parent_inheritance(self):
        """Higher parent pressure should boost child pressure."""
        high_parent = pdr._compute_pressure(0.5, depth=1, parent_pressure=0.9)
        low_parent = pdr._compute_pressure(0.5, depth=1, parent_pressure=0.1)
        assert high_parent > low_parent


# ============================================================================
# Test: Censorship detection
# ============================================================================

class TestCensorshipDetection:
    def test_empty_text_not_censored(self):
        # Empty/None text is an error, not censorship — returns False
        assert pdr._is_censored_response("") is False

    def test_short_text_is_censored(self):
        assert pdr._is_censored_response("Access denied") is True

    def test_normal_text_not_censored(self):
        text = "This is a normal webpage with enough content to pass the length check " * 3
        assert pdr._is_censored_response(text) is False

    def test_multiple_keywords_detected(self):
        # PR #9's version requires >= 2 keyword matches for long text
        text = "403 Forbidden - Access denied to this resource. Verify you are human." + " " * 50
        assert pdr._is_censored_response(text) is True

    def test_single_keyword_not_censored(self):
        # Single keyword in long text is not enough
        text = "This page mentions captcha but is otherwise normal content with enough text." + " " * 50
        assert pdr._is_censored_response(text) is False

    def test_error_prefix_not_censored(self):
        text = "Fetch error: HTTP 403 for https://example.com"
        assert pdr._is_censored_response(text) is False

    def test_none_text_not_censored(self):
        # None is an error, not censorship — returns False
        assert pdr._is_censored_response(None) is False

    def test_whitespace_only_is_censored(self):
        assert pdr._is_censored_response("   \n\t  ") is True


# ============================================================================
# Test: Enhanced web fetch fallback chain
# ============================================================================

class TestEnhancedWebFetch:
    @pytest.mark.asyncio
    async def test_tier0_httpx_used_first(self):
        """Tier 0 (httpx) is the first fetch attempt."""
        good_text = "This is real page content with enough text to pass checks. " * 5
        with patch.object(pdr, "_fetch_via_httpx", new_callable=AsyncMock,
                          return_value=good_text):
            result = await pdr.enhanced_web_fetch("https://example.com")
            assert "Content from" in result
            pdr._fetch_via_httpx.assert_called_once()

    @pytest.mark.asyncio
    async def test_falls_through_tiers_on_failure(self):
        """If httpx returns censored content, should try Playwright next."""
        # Use censored content (not a 403/451 error) so Tier 1 is attempted
        censored = "Access denied. Verify you are human. Please try again later."
        good_text = "Real content from Playwright with enough text to pass. " * 5
        with patch.object(pdr, "_fetch_via_httpx", new_callable=AsyncMock,
                          return_value=censored), \
             patch.object(pdr, "_fetch_via_playwright", new_callable=AsyncMock,
                          return_value=good_text), \
             patch.object(pdr, "_PLAYWRIGHT_AVAILABLE", True):
            result = await pdr.enhanced_web_fetch("https://example.com")
            assert "Content from" in result

    @pytest.mark.asyncio
    async def test_error_returned_bare_when_all_tiers_fail(self):
        """When all tiers fail, the error is returned without Content wrapper."""
        error_text = "Fetch error: HTTP 500 for https://example.com"
        with patch.object(pdr, "_fetch_via_httpx", new_callable=AsyncMock,
                          return_value=error_text), \
             patch.object(pdr, "_PLAYWRIGHT_AVAILABLE", False), \
             patch.object(pdr, "_SELENIUM_AVAILABLE", False), \
             patch.object(pdr, "BRIGHT_DATA_API_KEY", ""), \
             patch.object(pdr, "OXYLABS_USERNAME", ""):
            result = await pdr.enhanced_web_fetch("https://example.com")
            assert result == error_text

    @pytest.mark.asyncio
    async def test_extract_info_in_result(self):
        """The extract_info instruction should be included in results."""
        good_text = "Real content here with enough length. " * 5
        with patch.object(pdr, "_fetch_via_httpx", new_callable=AsyncMock,
                          return_value=good_text):
            result = await pdr.enhanced_web_fetch("https://example.com", extract_info="Find prices")
            assert "Find prices" in result


# ============================================================================
# Test: Curated event formatting
# ============================================================================

class TestFormatCuratedEvent:
    def test_start_event(self):
        msg = pdr._format_curated_event_fallback({"type": "start", "question": "What is X?"})
        assert "Investigating" in msg
        assert "What is X?" in msg

    def test_finding_event(self):
        msg = pdr._format_curated_event_fallback({
            "type": "finding",
            "finding": "Nimesulide is banned in several EU countries",
            "conditions_count": 5,
            "depth": 1,
            "node_id": "n1",
            "question": "Is Nimesulide available?",
        })
        assert "5 findings" in msg
        assert "depth 1" in msg
        assert "Nimesulide is banned" in msg

    def test_finding_event_depth_zero(self):
        msg = pdr._format_curated_event_fallback({
            "type": "finding",
            "finding": "Root finding",
            "conditions_count": 3,
            "depth": 0,
            "node_id": "n1",
            "question": "q",
        })
        assert "depth 0" not in msg  # depth 0 should not show depth label

    def test_branch_event(self):
        msg = pdr._format_curated_event_fallback({
            "type": "branch",
            "parent_question": "Is X true?",
            "children_count": 3,
            "top_child": "What evidence supports X?",
            "depth": 2,
        })
        assert "3 follow-up questions" in msg
        assert "What evidence supports X?" in msg

    def test_branch_event_single_child(self):
        msg = pdr._format_curated_event_fallback({
            "type": "branch",
            "parent_question": "q",
            "children_count": 1,
            "top_child": "sub-q",
            "depth": 1,
        })
        assert "1 follow-up question" in msg
        assert "questions" not in msg  # singular

    def test_summary_event(self):
        msg = pdr._format_curated_event_fallback({
            "type": "summary",
            "nodes_explored": 12,
            "conditions_count": 47,
        })
        assert "12 branches explored" in msg
        assert "47 findings" in msg

    def test_unknown_event_returns_empty(self):
        msg = pdr._format_curated_event_fallback({"type": "internal_debug"})
        assert msg == ""

    def test_empty_event_returns_empty(self):
        msg = pdr._format_curated_event_fallback({})
        assert msg == ""


# ============================================================================
# Test: Spawn sub-questions (mocked LLM)
# ============================================================================

class TestSpawnSubQuestions:
    @pytest.mark.asyncio
    async def test_spawns_children_from_llm_response(self):
        """LLM returns sub-questions -> ResearchNode children created."""
        node = pdr.ResearchNode(
            id="root", question="Is X safe?", context="Root",
            depth=0, pressure=0.9,
        )
        conditions = [
            pdr.AtomicCondition(fact="X has side effects in some cases", confidence=0.7, angle="root"),
            pdr.AtomicCondition(fact="X is approved in EU", confidence=0.9, angle="root"),
        ]
        llm_response = {
            "content": '{"sub_questions": [{"question": "What side effects does X have?", "context": "Need to verify safety claim", "pressure": 0.8}, {"question": "In which EU countries is X approved?", "context": "Verify regulatory status", "pressure": 0.6}]}'
        }

        with patch.object(pdr, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            children = await pdr._spawn_sub_questions(
                node, conditions, "Is X safe?", ["Is X safe?"], "req-1",
            )

        assert len(children) == 2
        assert children[0].question == "What side effects does X have?"
        assert children[0].parent_id == "root"
        assert children[0].depth == 1
        assert children[1].question == "In which EU countries is X approved?"

    @pytest.mark.asyncio
    async def test_respects_depth_limit(self):
        """Nodes at max depth should not spawn children."""
        node = pdr.ResearchNode(
            id="deep", question="q", context="c",
            depth=pdr.TREE_MAX_DEPTH, pressure=0.5,
        )
        conditions = [pdr.AtomicCondition(fact="fact", confidence=0.5, angle="a")]

        children = await pdr._spawn_sub_questions(node, conditions, "q", [], "req-1")
        assert children == []

    @pytest.mark.asyncio
    async def test_no_conditions_no_spawn(self):
        """No conditions -> no children."""
        node = pdr.ResearchNode(id="n", question="q", context="c", depth=0, pressure=0.5)
        children = await pdr._spawn_sub_questions(node, [], "q", [], "req-1")
        assert children == []

    @pytest.mark.asyncio
    async def test_filters_duplicate_questions(self):
        """Questions that overlap with existing ones should be filtered out."""
        node = pdr.ResearchNode(id="n", question="q", context="c", depth=0, pressure=0.9)
        conditions = [pdr.AtomicCondition(fact="fact", confidence=0.5, angle="a")]
        llm_response = {
            "content": '{"sub_questions": [{"question": "Is X safe?", "context": "duplicate", "pressure": 0.8}]}'
        }

        with patch.object(pdr, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            children = await pdr._spawn_sub_questions(
                node, conditions, "q", ["Is X safe?"], "req-1",
            )

        assert children == []

    @pytest.mark.asyncio
    async def test_filters_low_pressure_questions(self):
        """Questions below TREE_PRESSURE_THRESHOLD should be pruned."""
        node = pdr.ResearchNode(id="n", question="q", context="c", depth=0, pressure=0.5)
        conditions = [pdr.AtomicCondition(fact="fact", confidence=0.5, angle="a")]
        llm_response = {
            "content": '{"sub_questions": [{"question": "Minor trivia?", "context": "not important", "pressure": 0.01}]}'
        }

        with patch.object(pdr, "call_llm", new_callable=AsyncMock, return_value=llm_response):
            children = await pdr._spawn_sub_questions(
                node, conditions, "q", [], "req-1",
            )

        assert children == []

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        """LLM error should return empty list, not crash."""
        node = pdr.ResearchNode(id="n", question="q", context="c", depth=0, pressure=0.5)
        conditions = [pdr.AtomicCondition(fact="fact", confidence=0.5, angle="a")]

        with patch.object(pdr, "call_llm", new_callable=AsyncMock, return_value={"error": "timeout"}):
            children = await pdr._spawn_sub_questions(node, conditions, "q", [], "req-1")

        assert children == []

    @pytest.mark.asyncio
    async def test_handles_malformed_json(self):
        """Malformed LLM output should return empty list, not crash."""
        node = pdr.ResearchNode(id="n", question="q", context="c", depth=0, pressure=0.5)
        conditions = [pdr.AtomicCondition(fact="fact", confidence=0.5, angle="a")]

        with patch.object(pdr, "call_llm", new_callable=AsyncMock,
                          return_value={"content": "This is not JSON at all"}):
            children = await pdr._spawn_sub_questions(node, conditions, "q", [], "req-1")

        assert children == []


# ============================================================================
# Test: Research single node
# ============================================================================

class TestResearchSingleNode:
    @pytest.mark.asyncio
    async def test_feeds_conditions_to_collector(self):
        """Conditions from run_subagent should flow to the collector."""
        node = pdr.ResearchNode(id="n1", question="q", context="c", depth=0, pressure=0.8)
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        mock_result = pdr.SubagentResult(
            angle="q",
            conditions=[
                pdr.AtomicCondition(fact="Found something", confidence=0.8, angle="q"),
            ],
            turns_used=3,
            tool_calls_made=5,
        )

        with patch.object(pdr, "run_subagent", new_callable=AsyncMock, return_value=mock_result):
            conditions, sa_result = await pdr._research_single_node(
                node, "user query", "req-1", collector, curated_queue,
            )

        assert len(conditions) == 1
        assert conditions[0].fact == "Found something"
        all_conds = await collector.all_conditions()
        assert len(all_conds) == 1

    @pytest.mark.asyncio
    async def test_emits_curated_finding(self):
        """A curated event should be emitted when conditions are found."""
        node = pdr.ResearchNode(id="n1", question="q", context="c", depth=1, pressure=0.7)
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        mock_result = pdr.SubagentResult(
            angle="q",
            conditions=[
                pdr.AtomicCondition(fact="Important finding", confidence=0.9, angle="q"),
            ],
            turns_used=2,
            tool_calls_made=3,
        )

        with patch.object(pdr, "run_subagent", new_callable=AsyncMock, return_value=mock_result):
            await pdr._research_single_node(node, "user query", "req-1", collector, curated_queue)

        assert not curated_queue.empty()
        event = curated_queue.get_nowait()
        assert event["type"] == "finding"
        assert event["depth"] == 1

    @pytest.mark.asyncio
    async def test_no_curated_event_on_empty_conditions(self):
        """No curated event when subagent finds nothing."""
        node = pdr.ResearchNode(id="n1", question="q", context="c", depth=0, pressure=0.5)
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        mock_result = pdr.SubagentResult(angle="q", conditions=[], turns_used=2, tool_calls_made=1)

        with patch.object(pdr, "run_subagent", new_callable=AsyncMock, return_value=mock_result):
            await pdr._research_single_node(node, "user query", "req-1", collector, curated_queue)

        assert curated_queue.empty()


# ============================================================================
# Test: LangGraph state and graph topology
# ============================================================================

class TestPersistentResearchState:
    def test_state_has_nodes_explored(self):
        """PersistentResearchState should have nodes_explored field."""
        annotations = pdr.PersistentResearchState.__annotations__
        assert "nodes_explored" in annotations
        assert "nodes_explored" in annotations

    def test_state_no_angles_field(self):
        """PersistentResearchState should NOT have an 'angles' field anymore."""
        annotations = pdr.PersistentResearchState.__annotations__
        assert "angles" not in annotations


class TestGraphTopology:
    def test_graph_compiles(self):
        """The persistent research graph should compile without errors."""
        graph = pdr.build_persistent_research_graph()
        assert graph is not None

    def test_graph_has_tree_research_node(self):
        """Graph should have 'tree_research' node, not 'plan' or 'subagents'."""
        graph = pdr.build_persistent_research_graph()
        node_names = set(graph.get_graph().nodes.keys())
        assert "tree_research" in node_names
        assert "plan" not in node_names
        assert "subagents" not in node_names

    def test_graph_has_expected_nodes(self):
        """Graph should have all expected pipeline nodes."""
        graph = pdr.build_persistent_research_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"retrieve", "tree_research", "entities", "verify",
                    "reflect", "persist", "synthesize", "__start__", "__end__"}
        assert expected.issubset(node_names)


# ============================================================================
# Test: tree_research_reactor integration (mocked subagent)
# ============================================================================

def _empty_comprehension():
    """Return a QueryComprehension with no entities/domains so tests don't
    get polluted by understanding conditions admitted at reactor startup."""
    return pdr.QueryComprehension(
        entities=[], domains=[], implicit_questions=[],
        adjacent_territories=[], relevance_keywords=[],
        deep_knowledge_targets=[], semantic_summary="",
    )


def _make_mock_research_node(conditions, turns=1, tools=1):
    """Helper: create a mock _research_single_node coroutine result."""
    sa_result = pdr.SubagentResult(
        angle="q",
        conditions=conditions,
        turns_used=turns,
        tool_calls_made=tools,
    )
    return conditions, sa_result


class TestTreeResearchReactor:
    @pytest.mark.asyncio
    async def test_explores_root_and_produces_conditions(self):
        """Reactor should explore at least the root node and return conditions."""
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        conditions = [pdr.AtomicCondition(fact="Root finding", confidence=0.8, angle="root")]
        mock_node_result = _make_mock_research_node(conditions, turns=2, tools=3)

        original_timeout = pdr.TREE_WORKER_IDLE_TIMEOUT
        pdr.TREE_WORKER_IDLE_TIMEOUT = 1.0
        try:
            with patch.object(pdr, "comprehend_query", new_callable=AsyncMock,
                              return_value=_empty_comprehension()), \
                 patch.object(pdr, "_research_single_node", new_callable=AsyncMock,
                              return_value=mock_node_result), \
                 patch.object(pdr, "_spawn_sub_questions", new_callable=AsyncMock, return_value=[]):
                result = await pdr.tree_research_reactor(
                    user_query="What is X?",
                    prior_conditions=[],
                    graph_neighbors=[],
                    req_id="test-reactor",
                    collector=collector,
                    curated_queue=curated_queue,
                )
        finally:
            pdr.TREE_WORKER_IDLE_TIMEOUT = original_timeout

        assert len(result["all_conditions"]) >= 1
        assert result["all_conditions"][0].fact == "Root finding"
        assert result["total_turns"] == 2
        assert result["total_tools"] == 3
        assert len(result["progress_log"]) >= 1

    @pytest.mark.asyncio
    async def test_spawns_children_and_explores_them(self):
        """Reactor should explore root, spawn children, and explore those too."""
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        call_count = [0]

        async def mock_research_node(node, user_query, req_id, coll, cq, **kwargs):
            call_count[0] += 1
            conditions = [
                pdr.AtomicCondition(
                    fact=f"Finding from call {call_count[0]}",
                    confidence=0.7, angle=node.question,
                ),
            ]
            sa_result = pdr.SubagentResult(
                angle=node.question,
                conditions=conditions,
                turns_used=1,
                tool_calls_made=2,
            )
            return conditions, sa_result

        spawn_count = [0]

        async def mock_spawn(node, conditions, user_query, existing_questions, req_id, **kwargs):
            spawn_count[0] += 1
            if spawn_count[0] == 1:
                # Only root spawns children
                return [
                    pdr.ResearchNode(
                        id=f"child-{i}",
                        question=f"Sub-question {i}?",
                        context="Follow-up",
                        depth=node.depth + 1,
                        pressure=0.6,
                        parent_id=node.id,
                    )
                    for i in range(2)
                ]
            return []

        original_timeout = pdr.TREE_WORKER_IDLE_TIMEOUT
        # Workers need enough idle time to wait for root to finish and
        # spawn children before they time out.
        pdr.TREE_WORKER_IDLE_TIMEOUT = 5.0
        try:
            with patch.object(pdr, "comprehend_query", new_callable=AsyncMock,
                              return_value=_empty_comprehension()), \
                 patch.object(pdr, "_research_single_node", side_effect=mock_research_node), \
                 patch.object(pdr, "_spawn_sub_questions", side_effect=mock_spawn):
                result = await pdr.tree_research_reactor(
                    user_query="What is X?",
                    prior_conditions=[],
                    graph_neighbors=[],
                    req_id="test-reactor-2",
                    collector=collector,
                    curated_queue=curated_queue,
                )
        finally:
            pdr.TREE_WORKER_IDLE_TIMEOUT = original_timeout

        # Root + 2 children = 3 explorations
        assert len(result["all_conditions"]) == 3
        assert len(result["subagent_results"]) == 3

    @pytest.mark.asyncio
    async def test_respects_node_budget(self):
        """Reactor should not exceed TREE_MAX_NODES."""
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        async def mock_research_node(node, user_query, req_id, coll, cq, **kwargs):
            conditions = [pdr.AtomicCondition(fact="f", confidence=0.5, angle="a")]
            sa = pdr.SubagentResult(angle="q", conditions=conditions, turns_used=1, tool_calls_made=1)
            return conditions, sa

        async def mock_spawn(node, conditions, user_query, existing_questions, req_id, **kwargs):
            # Always try to spawn 5 children
            return [
                pdr.ResearchNode(
                    id=f"n-{uuid.uuid4().hex[:6]}",
                    question=f"Q at depth {node.depth + 1}",
                    context="c",
                    depth=node.depth + 1,
                    pressure=0.5,
                    parent_id=node.id,
                )
                for _ in range(5)
            ]

        # Set a small budget and short timeout for the test
        original_max = pdr.TREE_MAX_NODES
        original_timeout = pdr.TREE_WORKER_IDLE_TIMEOUT
        pdr.TREE_MAX_NODES = 8
        pdr.TREE_WORKER_IDLE_TIMEOUT = 1.0
        try:
            with patch.object(pdr, "comprehend_query", new_callable=AsyncMock,
                              return_value=_empty_comprehension()), \
                 patch.object(pdr, "_research_single_node", side_effect=mock_research_node), \
                 patch.object(pdr, "_spawn_sub_questions", side_effect=mock_spawn):
                result = await pdr.tree_research_reactor(
                    user_query="q",
                    prior_conditions=[],
                    graph_neighbors=[],
                    req_id="test-budget",
                    collector=collector,
                    curated_queue=curated_queue,
                )

            assert len(result["subagent_results"]) <= 8
        finally:
            pdr.TREE_MAX_NODES = original_max
            pdr.TREE_WORKER_IDLE_TIMEOUT = original_timeout

    @pytest.mark.asyncio
    async def test_curated_queue_receives_events(self):
        """The curated queue should get start and summary events."""
        collector = pdr.LiveFindingsCollector()
        curated_queue: asyncio.Queue = asyncio.Queue()

        conditions = [pdr.AtomicCondition(fact="f", confidence=0.7, angle="a")]
        mock_node_result = _make_mock_research_node(conditions)

        original_timeout = pdr.TREE_WORKER_IDLE_TIMEOUT
        pdr.TREE_WORKER_IDLE_TIMEOUT = 1.0
        try:
            with patch.object(pdr, "comprehend_query", new_callable=AsyncMock,
                              return_value=_empty_comprehension()), \
                 patch.object(pdr, "_research_single_node", new_callable=AsyncMock,
                              return_value=mock_node_result), \
                 patch.object(pdr, "_spawn_sub_questions", new_callable=AsyncMock, return_value=[]):
                await pdr.tree_research_reactor(
                    user_query="What is X?",
                    prior_conditions=[],
                    graph_neighbors=[],
                    req_id="test-events",
                    collector=collector,
                    curated_queue=curated_queue,
                )
        finally:
            pdr.TREE_WORKER_IDLE_TIMEOUT = original_timeout

        events = []
        while not curated_queue.empty():
            events.append(curated_queue.get_nowait())

        event_types = [e["type"] for e in events]
        assert "start" in event_types
        assert "summary" in event_types


# ============================================================================
# Regression: old planning path is dead, tree reactor is the active path
# ============================================================================

class TestOldPlanningPathIsDead:
    """Regression tests ensuring the old plan-10-angles + parallel-subagents
    path is dead code and the tree reactor is the sole research pipeline."""

    def test_plan_research_not_called_by_any_graph_node(self):
        """plan_research() should exist but not be referenced by any graph node."""
        import inspect
        graph_node_funcs = [
            pdr.pdr_node_retrieve,
            pdr.pdr_node_tree_research,
            pdr.pdr_node_entities,
            pdr.pdr_node_verify,
            pdr.pdr_node_reflect,
            pdr.pdr_node_persist,
            pdr.pdr_node_synthesize,
        ]
        for func in graph_node_funcs:
            source = inspect.getsource(func)
            assert "plan_research(" not in source, (
                f"{func.__name__} still calls plan_research() — "
                f"the old planning path should be dead"
            )

    def test_run_subagent_not_called_by_any_graph_node(self):
        """run_subagent() (old parallel dispatch) should not be called by graph nodes."""
        import inspect
        graph_node_funcs = [
            pdr.pdr_node_retrieve,
            pdr.pdr_node_tree_research,
            pdr.pdr_node_entities,
            pdr.pdr_node_verify,
            pdr.pdr_node_reflect,
            pdr.pdr_node_persist,
            pdr.pdr_node_synthesize,
        ]
        for func in graph_node_funcs:
            source = inspect.getsource(func)
            assert "run_subagent(" not in source, (
                f"{func.__name__} still calls run_subagent() — "
                f"the old parallel dispatch should be dead"
            )

    def test_graph_edge_topology_is_linear_tree_pipeline(self):
        """Graph edges should follow: retrieve → tree_research → entities → verify
        → reflect → persist → synthesize → END. No plan/subagent nodes."""
        graph = pdr.build_persistent_research_graph()
        g = graph.get_graph()
        node_names = set(g.nodes.keys())

        # Must NOT contain old-style nodes
        for forbidden in ("plan", "subagents", "plan_angles", "dispatch"):
            assert forbidden not in node_names, (
                f"Graph still contains old node '{forbidden}'"
            )

        # Must contain the tree reactor
        assert "tree_research" in node_names

    def test_tree_research_calls_tree_research_reactor(self):
        """pdr_node_tree_research must delegate to tree_research_reactor."""
        import inspect
        source = inspect.getsource(pdr.pdr_node_tree_research)
        assert "tree_research_reactor(" in source, (
            "pdr_node_tree_research does not call tree_research_reactor — "
            "the tree reactor is not being used"
        )

    def test_state_has_no_legacy_fields(self):
        """PersistentResearchState should not have legacy planning fields."""
        annotations = pdr.PersistentResearchState.__annotations__
        for legacy_field in ("angles", "bridge_queries", "plan"):
            assert legacy_field not in annotations, (
                f"PersistentResearchState still has legacy field '{legacy_field}'"
            )
        # Should have tree reactor fields
        assert "nodes_explored" in annotations

    @pytest.mark.asyncio
    async def test_pdr_node_tree_research_invokes_reactor_not_planner(self):
        """Full integration: pdr_node_tree_research should call
        tree_research_reactor (not plan_research) when executed."""
        collector = pdr.LiveFindingsCollector()
        curated_queue = asyncio.Queue()

        # Pre-populate the module-level dicts that pdr_node_tree_research reads
        req_id = "test-regression-no-planner"
        pdr._live_collectors[req_id] = collector
        pdr._curated_queues[req_id] = curated_queue

        conditions = [pdr.AtomicCondition(fact="Tree finding", confidence=0.9, angle="root")]
        reactor_result = {
            "all_conditions": conditions,
            "subagent_results": [
                pdr.SubagentResult(angle="root", conditions=conditions, turns_used=1, tool_calls_made=1),
            ],
            "total_turns": 1,
            "total_tools": 1,
            "total_children": 0,
            "progress_log": ["Tree reactor ran"],
        }

        state = {
            "req_id": req_id,
            "user_query": "Test query",
            "prior_conditions": [],
            "graph_neighbors": [],
        }

        with patch.object(pdr, "tree_research_reactor", new_callable=AsyncMock,
                          return_value=reactor_result) as mock_reactor, \
             patch.object(pdr, "plan_research", new_callable=AsyncMock) as mock_planner:

            result = await pdr.pdr_node_tree_research(state)

            # tree_research_reactor MUST have been called
            mock_reactor.assert_called_once()
            # plan_research MUST NOT have been called
            mock_planner.assert_not_called()

        assert result["nodes_explored"] == 1
        assert result["all_conditions"][0].fact == "Tree finding"
        assert result["phase"] == "entities"

        # Cleanup
        pdr._live_collectors.pop(req_id, None)
        pdr._curated_queues.pop(req_id, None)

    @pytest.mark.asyncio
    async def test_tree_produces_branching_not_flat_angles(self):
        """Tree reactor should produce a tree structure (root + children at
        varying depths), not a flat list of N parallel angles."""
        collector = pdr.LiveFindingsCollector()
        curated_queue = asyncio.Queue()

        explored_depths = []

        async def mock_research_node(node, user_query, req_id, coll, cq, **kwargs):
            explored_depths.append(node.depth)
            conditions = [pdr.AtomicCondition(
                fact=f"Finding at depth {node.depth}", confidence=0.7, angle=node.question,
            )]
            sa = pdr.SubagentResult(
                angle=node.question, conditions=conditions,
                turns_used=1, tool_calls_made=1,
            )
            return conditions, sa

        spawn_count = [0]

        async def mock_spawn(node, conditions, user_query, existing_questions, req_id, **kwargs):
            spawn_count[0] += 1
            if node.depth == 0:
                # Root spawns 2 children
                return [
                    pdr.ResearchNode(
                        id=f"child-{i}", question=f"Follow-up {i}?",
                        context="c", depth=1, pressure=0.7, parent_id=node.id,
                    )
                    for i in range(2)
                ]
            elif node.depth == 1 and spawn_count[0] <= 2:
                # First child spawns 1 grandchild
                return [
                    pdr.ResearchNode(
                        id=f"grandchild-{node.id}", question=f"Deep follow-up?",
                        context="c", depth=2, pressure=0.5, parent_id=node.id,
                    ),
                ]
            return []

        original_timeout = pdr.TREE_WORKER_IDLE_TIMEOUT
        pdr.TREE_WORKER_IDLE_TIMEOUT = 5.0
        try:
            with patch.object(pdr, "comprehend_query", new_callable=AsyncMock,
                              return_value=_empty_comprehension()), \
                 patch.object(pdr, "_research_single_node", side_effect=mock_research_node), \
                 patch.object(pdr, "_spawn_sub_questions", side_effect=mock_spawn):
                result = await pdr.tree_research_reactor(
                    user_query="Test branching",
                    prior_conditions=[],
                    graph_neighbors=[],
                    req_id="test-branching",
                    collector=collector,
                    curated_queue=curated_queue,
                )
        finally:
            pdr.TREE_WORKER_IDLE_TIMEOUT = original_timeout

        # Should have explored nodes at multiple depths (tree, not flat)
        assert 0 in explored_depths, "Root (depth 0) was not explored"
        assert 1 in explored_depths, "Children (depth 1) were not explored"
        assert 2 in explored_depths, "Grandchildren (depth 2) were not explored"

        # NOT a flat list of 10 — should have root + 2 children + at least 1 grandchild
        assert len(result["subagent_results"]) >= 4
        # Verify multiple depth levels were explored (tree, not flat dispatch)
        unique_depths = set(explored_depths)
        assert len(unique_depths) >= 3, (
            f"Expected at least 3 depth levels (0, 1, 2), got {unique_depths}"
        )


# ============================================================================
# Test: _pdr_append_log reducer
# ============================================================================

class TestPdrAppendLog:
    def test_merges_lists(self):
        result = pdr._pdr_append_log(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]

    def test_empty_left(self):
        assert pdr._pdr_append_log([], ["x"]) == ["x"]

    def test_empty_right(self):
        assert pdr._pdr_append_log(["x"], []) == ["x"]

    def test_both_empty(self):
        assert pdr._pdr_append_log([], []) == []
