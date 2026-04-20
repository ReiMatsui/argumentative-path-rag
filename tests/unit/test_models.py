"""コアデータモデルのユニットテスト。"""

import pytest
from pydantic import ValidationError

from ap_rag.models import (
    ArgumentEdge,
    ArgumentGraph,
    ArgumentNode,
    EdgeType,
    NodeType,
    QueryType,
    TRAVERSAL_STRATEGIES,
)


# ── ArgumentNode ──────────────────────────────────────────────────────────────

class TestArgumentNode:
    def _make(self, **kwargs) -> ArgumentNode:
        defaults = dict(
            node_type=NodeType.CLAIM,
            text="Q3売上は前年比12%減少した。",
            source_doc_id="doc_001",
            source_chunk_idx=0,
        )
        return ArgumentNode(**{**defaults, **kwargs})

    def test_normal_creation(self):
        node = self._make()
        assert node.node_type == NodeType.CLAIM
        assert node.id  # UUID が自動生成されている

    def test_text_is_stripped(self):
        node = self._make(text="  hello  ")
        assert node.text == "hello"

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            self._make(text="   ")

    def test_immutable(self):
        node = self._make()
        with pytest.raises(Exception):  # frozen=True なので AttributeError or ValidationError
            node.text = "changed"  # type: ignore

    def test_negative_chunk_idx_raises(self):
        with pytest.raises(ValidationError):
            self._make(source_chunk_idx=-1)


# ── ArgumentEdge ──────────────────────────────────────────────────────────────

class TestArgumentEdge:
    def test_self_loop_raises(self):
        with pytest.raises(ValidationError):
            ArgumentEdge(
                edge_type=EdgeType.SUPPORTS,
                source_id="same",
                target_id="same",
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            ArgumentEdge(
                edge_type=EdgeType.SUPPORTS,
                source_id="a",
                target_id="b",
                confidence=1.5,
            )

    def test_normal_creation(self):
        edge = ArgumentEdge(
            edge_type=EdgeType.SUPPORTS,
            source_id="node_a",
            target_id="node_b",
            confidence=0.9,
        )
        assert edge.edge_type == EdgeType.SUPPORTS


# ── ArgumentGraph ─────────────────────────────────────────────────────────────

def _make_node(node_type: NodeType = NodeType.CLAIM, idx: int = 0) -> ArgumentNode:
    return ArgumentNode(
        node_type=node_type,
        text=f"テキスト {idx}",
        source_doc_id="doc_001",
        source_chunk_idx=idx,
    )


class TestArgumentGraph:
    def test_add_and_retrieve_node(self):
        g = ArgumentGraph(doc_id="doc_001")
        node = _make_node()
        g.add_node(node)
        assert g.get_node(node.id) == node

    def test_duplicate_node_raises(self):
        g = ArgumentGraph(doc_id="doc_001")
        node = _make_node()
        g.add_node(node)
        with pytest.raises(ValueError, match="既に存在"):
            g.add_node(node)

    def test_add_edge_with_missing_nodes_raises(self):
        g = ArgumentGraph(doc_id="doc_001")
        with pytest.raises(ValueError, match="始点ノード"):
            g.add_edge(ArgumentEdge(
                edge_type=EdgeType.SUPPORTS,
                source_id="ghost",
                target_id="ghost2",
            ))

    def test_outgoing_and_incoming_edges(self):
        g = ArgumentGraph(doc_id="doc_001")
        ev = _make_node(NodeType.EVIDENCE, 0)
        cl = _make_node(NodeType.CLAIM, 1)
        g.add_node(ev)
        g.add_node(cl)
        edge = ArgumentEdge(
            edge_type=EdgeType.SUPPORTS,
            source_id=ev.id,
            target_id=cl.id,
        )
        g.add_edge(edge)

        assert g.outgoing_edges(ev.id) == [edge]
        assert g.incoming_edges(cl.id) == [edge]
        assert g.outgoing_edges(cl.id) == []

    def test_stats(self):
        g = ArgumentGraph(doc_id="doc_001")
        g.add_node(_make_node(NodeType.CLAIM, 0))
        g.add_node(_make_node(NodeType.EVIDENCE, 1))
        stats = g.stats()
        assert stats["total_nodes"] == 2
        assert stats["nodes_by_type"]["CLAIM"] == 1
        assert stats["nodes_by_type"]["EVIDENCE"] == 1


# ── TraversalStrategy ─────────────────────────────────────────────────────────

class TestTraversalStrategy:
    def test_all_query_types_have_strategy(self):
        for qt in QueryType:
            assert qt in TRAVERSAL_STRATEGIES, f"{qt} に探索戦略が定義されていません"

    def test_why_excludes_contrast(self):
        strategy = TRAVERSAL_STRATEGIES[QueryType.WHY]
        assert NodeType.CONTRAST in strategy.exclude_node_types

    def test_what_has_no_traversal(self):
        strategy = TRAVERSAL_STRATEGIES[QueryType.WHAT]
        assert strategy.follow_edges == []
        assert strategy.max_depth == 0
