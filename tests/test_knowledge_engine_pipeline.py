"""Tests for the knowledge engine ingest pipeline LangGraph.

Verifies:
- The 7-node linear graph topology executes all steps in order
- Each node produces the expected state updates
- The _etl_append_log reducer merges progress entries correctly
- Individual nodes handle edge cases (empty text, rebuild vs append mode)

All external dependencies (Neo4j, LLM calls, filesystem) are mocked.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Add the knowledge engine package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "knowledge-engine"))

from knowledge_engine.pipeline import (
    _etl_append_log,
    IngestPipelineState,
    etl_node_save_raw,
    etl_node_clear_namespace,
    etl_node_chunk,
    etl_node_extract,
    etl_node_load,
    etl_node_resolve,
    etl_node_graph_metrics,
    build_ingest_pipeline_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_pipeline_state(**overrides) -> dict:
    """Build a minimal IngestPipelineState dict for testing."""
    state: dict = {
        "job_id": "test-job-001",
        "namespace": "test-ns",
        "doc_id": "doc-abc123",
        "title": "Test Document",
        "text": "This is a test document with enough text to be meaningful. " * 20,
        "source": "test-source",
        "rebuild": False,
        "raw_path": "",
        "chunks": [],
        "extraction_results": [],
        "load_stats": {},
        "resolve_stats": {},
        "progress_log": [],
        "phase": "save_raw",
    }
    state.update(overrides)
    return state


# ============================================================================
# Test: _etl_append_log reducer
# ============================================================================

class TestEtlAppendLog:
    def test_merges_lists(self):
        assert _etl_append_log(["a"], ["b", "c"]) == ["a", "b", "c"]

    def test_empty_left(self):
        assert _etl_append_log([], ["x"]) == ["x"]

    def test_empty_right(self):
        assert _etl_append_log(["x"], []) == ["x"]

    def test_both_empty(self):
        assert _etl_append_log([], []) == []


# ============================================================================
# Test: etl_node_save_raw
# ============================================================================

class TestEtlNodeSaveRaw:
    @pytest.mark.asyncio
    async def test_saves_raw_file(self, tmp_path):
        state = _base_pipeline_state()
        with patch("knowledge_engine.pipeline._save_raw_file", return_value=str(tmp_path / "raw.txt")) as mock_save:
            result = await etl_node_save_raw(state)

        mock_save.assert_called_once_with(
            state["namespace"], state["doc_id"], state["title"], state["text"],
        )
        assert result["raw_path"] == str(tmp_path / "raw.txt")
        assert result["phase"] == "clear_namespace"
        assert len(result["progress_log"]) == 1
        assert "Saved raw file" in result["progress_log"][0]


# ============================================================================
# Test: etl_node_clear_namespace
# ============================================================================

class TestEtlNodeClearNamespace:
    @pytest.mark.asyncio
    async def test_rebuild_clears_namespace(self):
        state = _base_pipeline_state(rebuild=True)
        with patch("knowledge_engine.pipeline.clear_namespace", return_value=42) as mock_clear:
            result = await etl_node_clear_namespace(state)

        mock_clear.assert_called_once_with("test-ns")
        assert result["phase"] == "chunk"
        assert "42 nodes removed" in result["progress_log"][0]

    @pytest.mark.asyncio
    async def test_append_mode_skips_clear(self):
        state = _base_pipeline_state(rebuild=False)
        with patch("knowledge_engine.pipeline.clear_namespace") as mock_clear:
            result = await etl_node_clear_namespace(state)

        mock_clear.assert_not_called()
        assert result["phase"] == "chunk"
        assert "skipping" in result["progress_log"][0].lower()


# ============================================================================
# Test: etl_node_chunk
# ============================================================================

class TestEtlNodeChunk:
    @pytest.mark.asyncio
    async def test_chunks_text(self):
        state = _base_pipeline_state(text="Hello world. " * 500)
        with patch("knowledge_engine.pipeline.chunk_text") as mock_chunk:
            mock_chunk.return_value = [
                {"id": "c1", "chunk_index": 0, "content": "chunk1"},
                {"id": "c2", "chunk_index": 1, "content": "chunk2"},
            ]
            result = await etl_node_chunk(state)

        assert len(result["chunks"]) == 2
        assert result["phase"] == "extract"
        assert "2 chunks" in result["progress_log"][0]


# ============================================================================
# Test: etl_node_extract
# ============================================================================

class TestEtlNodeExtract:
    @pytest.mark.asyncio
    async def test_extracts_from_chunks(self):
        state = _base_pipeline_state(
            chunks=[
                {"id": "c1", "chunk_index": 0, "content": "chunk1"},
                {"id": "c2", "chunk_index": 1, "content": "chunk2"},
            ],
        )
        mock_results = [
            {"chunk_id": "c1", "concepts": ["A"], "claims": []},
            {"chunk_id": "c2", "concepts": ["B"], "claims": []},
        ]
        with patch("knowledge_engine.pipeline.extract_all_chunks", return_value=mock_results) as mock_extract:
            result = await etl_node_extract(state)

        mock_extract.assert_called_once_with(state["chunks"])
        assert len(result["extraction_results"]) == 2
        assert result["phase"] == "load"
        assert "3 passes" in result["progress_log"][0]


# ============================================================================
# Test: etl_node_load
# ============================================================================

class TestEtlNodeLoad:
    @pytest.mark.asyncio
    async def test_loads_into_neo4j(self):
        state = _base_pipeline_state(
            chunks=[{"id": "c1", "chunk_index": 0, "content": "chunk1"}],
            extraction_results=[{"chunk_id": "c1", "concepts": ["A"]}],
        )
        mock_stats = {"nodes_created": 10, "relationships_created": 5}
        with patch("knowledge_engine.pipeline._load_into_neo4j", return_value=mock_stats) as mock_load:
            result = await etl_node_load(state)

        mock_load.assert_called_once_with(
            namespace="test-ns",
            doc_id="doc-abc123",
            title="Test Document",
            source="test-source",
            chunks=state["chunks"],
            extraction_results=state["extraction_results"],
        )
        assert result["load_stats"] == mock_stats
        assert result["phase"] == "resolve"


# ============================================================================
# Test: etl_node_resolve
# ============================================================================

class TestEtlNodeResolve:
    @pytest.mark.asyncio
    async def test_resolves_entities(self):
        state = _base_pipeline_state()
        mock_stats = {"merges": 3, "total_concepts": 10}
        with patch("knowledge_engine.pipeline.resolve_entities", return_value=mock_stats) as mock_resolve:
            result = await etl_node_resolve(state)

        mock_resolve.assert_called_once_with("test-ns")
        assert result["resolve_stats"] == mock_stats
        assert result["phase"] == "graph_metrics"


# ============================================================================
# Test: etl_node_graph_metrics
# ============================================================================

class TestEtlNodeGraphMetrics:
    @pytest.mark.asyncio
    async def test_computes_metrics(self):
        state = _base_pipeline_state()
        with patch("knowledge_engine.pipeline._compute_graph_metrics") as mock_metrics:
            result = await etl_node_graph_metrics(state)

        mock_metrics.assert_called_once_with("test-ns")
        assert result["phase"] == "done"
        assert "community detection" in result["progress_log"][0].lower()


# ============================================================================
# Test: build_ingest_pipeline_graph topology
# ============================================================================

class TestPipelineGraphTopology:
    def test_graph_compiles(self):
        """The graph should compile without error."""
        graph = build_ingest_pipeline_graph()
        assert graph is not None

    def test_graph_has_correct_nodes(self):
        """The compiled graph should contain all 7 pipeline nodes."""
        graph = build_ingest_pipeline_graph()
        # LangGraph compiled graphs expose nodes via .nodes
        node_names = set(graph.nodes.keys())
        expected_nodes = {
            "save_raw", "clear_namespace", "chunk", "extract",
            "load", "resolve", "graph_metrics",
            "__start__", "__end__",
        }
        # Check that all pipeline nodes exist (graph may have __start__/__end__)
        pipeline_nodes = {"save_raw", "clear_namespace", "chunk", "extract",
                          "load", "resolve", "graph_metrics"}
        assert pipeline_nodes.issubset(node_names), (
            f"Missing nodes: {pipeline_nodes - node_names}"
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_runs_end_to_end(self):
        """Run the full pipeline with all external deps mocked.
        Verify that all 7 nodes execute in order and produce correct
        final state.
        """
        graph = build_ingest_pipeline_graph()

        initial_state: dict = {
            "job_id": "test-e2e",
            "namespace": "test-ns",
            "doc_id": "doc-e2e",
            "title": "E2E Test",
            "text": "Test content for end to end pipeline verification.",
            "source": "unit-test",
            "rebuild": True,
            "raw_path": "",
            "chunks": [],
            "extraction_results": [],
            "load_stats": {},
            "resolve_stats": {},
            "progress_log": [],
            "phase": "save_raw",
        }

        mock_chunks = [
            {"id": "c1", "chunk_index": 0, "content": "Test content for end to end pipeline verification."},
        ]
        mock_extraction = [
            {"chunk_id": "c1", "concepts": ["pipeline", "testing"], "claims": []},
        ]
        mock_load_stats = {"nodes_created": 5, "relationships_created": 2}
        mock_resolve_stats = {"merges": 0, "total_concepts": 2}

        with patch("knowledge_engine.pipeline._save_raw_file", return_value="/tmp/test_raw.txt"), \
             patch("knowledge_engine.pipeline.clear_namespace", return_value=10), \
             patch("knowledge_engine.pipeline.chunk_text", return_value=mock_chunks), \
             patch("knowledge_engine.pipeline.extract_all_chunks", return_value=mock_extraction), \
             patch("knowledge_engine.pipeline._load_into_neo4j", return_value=mock_load_stats), \
             patch("knowledge_engine.pipeline.resolve_entities", return_value=mock_resolve_stats), \
             patch("knowledge_engine.pipeline._compute_graph_metrics"):

            config = {"configurable": {"thread_id": "test-e2e"}}
            final_state = None
            phases_seen = []

            async for state_update in graph.astream(
                initial_state, config=config, stream_mode="values",
            ):
                phase = state_update.get("phase", "")
                if phase and phase not in phases_seen:
                    phases_seen.append(phase)
                final_state = state_update

        assert final_state is not None
        assert final_state["phase"] == "done"
        assert final_state["raw_path"] == "/tmp/test_raw.txt"
        assert len(final_state["chunks"]) == 1
        assert len(final_state["extraction_results"]) == 1
        assert final_state["load_stats"] == mock_load_stats
        assert final_state["resolve_stats"] == mock_resolve_stats

        # Verify all 7 phases were seen (progress_log should have entries from all nodes)
        progress = final_state.get("progress_log", [])
        assert len(progress) >= 7, f"Expected >= 7 progress entries, got {len(progress)}: {progress}"

        # Verify phase progression
        expected_phases = [
            "save_raw", "clear_namespace", "chunk", "extract",
            "load", "resolve", "graph_metrics", "done",
        ]
        assert phases_seen == expected_phases, (
            f"Phase order mismatch: {phases_seen} != {expected_phases}"
        )
