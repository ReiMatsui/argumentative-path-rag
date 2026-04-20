"""検索エンジン各コンポーネントのユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ap_rag.retrieval.query_classifier import QueryClassifier
from ap_rag.retrieval.traversal import GraphTraverser
from ap_rag.retrieval.context_builder import ContextBuilder
from ap_rag.graph.store import GraphStore
from ap_rag.models import ArgumentNode, NodeType, QueryType
from ap_rag.models.taxonomy import TRAVERSAL_STRATEGIES


# ── QueryClassifier ───────────────────────────────────────────────────────────

class TestQueryClassifier:
    def _make_classifier(self, llm_response: str) -> QueryClassifier:
        mock_choice = MagicMock()
        mock_choice.message.content = llm_response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return QueryClassifier(client=mock_client)

    @pytest.mark.parametrize("llm_output,expected", [
        ("WHY", QueryType.WHY),
        ("WHAT", QueryType.WHAT),
        ("HOW", QueryType.HOW),
        ("EVIDENCE", QueryType.EVIDENCE),
        ("ASSUMPTION", QueryType.ASSUMPTION),
    ])
    def test_classify_known_types(self, llm_output: str, expected: QueryType):
        classifier = self._make_classifier(llm_output)
        assert classifier.classify("some query") == expected

    def test_unknown_response_falls_back_to_what(self):
        classifier = self._make_classifier("UNKNOWN")
        result = classifier.classify("some query")
        assert result == QueryType.WHAT

    def test_partial_match(self):
        classifier = self._make_classifier("I think this is WHY type")
        result = classifier.classify("some query")
        assert result == QueryType.WHY


# ── GraphTraverser ────────────────────────────────────────────────────────────

def _make_node(node_type: NodeType, text: str = "テキスト") -> ArgumentNode:
    return ArgumentNode(
        node_type=node_type,
        text=text,
        source_doc_id="doc_001",
        source_chunk_idx=0,
    )


class TestGraphTraverser:
    def _make_store(
        self,
        incoming_map: dict[str, list[ArgumentNode]] | None = None,
        outgoing_map: dict[str, list[ArgumentNode]] | None = None,
    ) -> GraphStore:
        """direction を考慮したモックストアを生成する。

        Args:
            incoming_map: direction="incoming" のときに返すノード（node_id → list）。
            outgoing_map: direction="outgoing" のときに返すノード（node_id → list）。
        """
        incoming_map = incoming_map or {}
        outgoing_map = outgoing_map or {}

        def _get_neighbors(node_id, edge_types=None, direction="incoming"):
            if direction == "incoming":
                return incoming_map.get(node_id, [])
            else:
                return outgoing_map.get(node_id, [])

        mock_store = MagicMock(spec=GraphStore)
        mock_store.get_neighbors.side_effect = _get_neighbors
        return mock_store

    def test_what_type_traverses_to_evidence(self):
        """WHAT 型は SUPPORTS(incoming) で depth=2 まで EVIDENCE を取得する。

        旧実装 (max_depth=0) では EVIDENCE が完全に無視されていた。
        v2 修正により WHAT も SUPPORTS/DERIVES 辺を辿って具体的根拠を取得する。
        """
        claim = _make_node(NodeType.CLAIM, "使用した特徴量はいくつか。")
        evidence = _make_node(NodeType.EVIDENCE, "特徴量 A, B, C を使用した。")
        store = self._make_store(incoming_map={claim.id: [evidence]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.WHAT)
        node_ids = {n.id for n in result}
        assert claim.id in node_ids
        assert evidence.id in node_ids  # EVIDENCE が取得される
        store.get_neighbors.assert_called()  # 探索が行われた

    def test_why_type_traverses_to_evidence(self):
        """WHY 型は SUPPORTS(incoming) で EVIDENCE ノードまで辿る。"""
        claim = _make_node(NodeType.CLAIM, "売上が減少した。")
        evidence = _make_node(NodeType.EVIDENCE, "在庫調整が主因。")
        store = self._make_store(incoming_map={claim.id: [evidence]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.WHY)
        node_ids = {n.id for n in result}
        assert claim.id in node_ids
        assert evidence.id in node_ids

    def test_why_type_traverses_to_assumption_via_outgoing(self):
        """WHY 型は ASSUMES(outgoing) で ASSUMPTION ノードまで辿る。"""
        claim = _make_node(NodeType.CLAIM, "売上が減少した。")
        assumption = _make_node(NodeType.ASSUMPTION, "為替は安定前提。")
        # ASSUMES は outgoing（CLAIM → ASSUMPTION）
        store = self._make_store(outgoing_map={claim.id: [assumption]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.WHY)
        node_ids = {n.id for n in result}
        assert claim.id in node_ids
        assert assumption.id in node_ids

    def test_why_type_excludes_contrast_nodes(self):
        """WHY 型は CONTRAST ノードを除外する。"""
        claim = _make_node(NodeType.CLAIM, "売上が減少した。")
        contrast = _make_node(NodeType.CONTRAST, "競合は増加した。")
        store = self._make_store(incoming_map={claim.id: [contrast]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.WHY)
        node_ids = {n.id for n in result}
        assert contrast.id not in node_ids

    def test_assumption_type_uses_outgoing(self):
        """ASSUMPTION 型は ASSUMES(outgoing) で ASSUMPTION ノードを取得する。"""
        claim = _make_node(NodeType.CLAIM, "売上が減少した。")
        assumption = _make_node(NodeType.ASSUMPTION, "半導体不足は構造的問題。")
        # ASSUMES は outgoing（CLAIM → ASSUMPTION）
        store = self._make_store(outgoing_map={claim.id: [assumption]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.ASSUMPTION)
        node_ids = {n.id for n in result}
        assert claim.id in node_ids
        assert assumption.id in node_ids

    def test_assumption_type_does_not_use_incoming(self):
        """ASSUMPTION 型で incoming 側にノードがあっても取得されない。"""
        claim = _make_node(NodeType.CLAIM, "売上が減少した。")
        evidence = _make_node(NodeType.EVIDENCE, "在庫調整が主因。")
        # incoming 側に EVIDENCE を置いても ASSUMPTION 型では無視される
        store = self._make_store(incoming_map={claim.id: [evidence]})
        traverser = GraphTraverser(store)
        result = traverser.traverse([claim], QueryType.ASSUMPTION)
        node_ids = {n.id for n in result}
        assert evidence.id not in node_ids

    def test_max_depth_limits_traversal(self):
        """depth=1 でA→B→Cのとき、AがエントリーならBまでしか辿らない。"""
        a = _make_node(NodeType.CLAIM, "A")
        b = _make_node(NodeType.EVIDENCE, "B")
        c = _make_node(NodeType.CONCLUSION, "C")
        store = self._make_store(incoming_map={a.id: [b], b.id: [c]})
        traverser = GraphTraverser(store)

        from ap_rag.models.taxonomy import TraversalStrategy, EdgeType
        strategy = TraversalStrategy(
            entry_node_types=[NodeType.CLAIM],
            follow_edges=[EdgeType.SUPPORTS],
            exclude_node_types=[],
            max_depth=1,
        )
        result = traverser._bfs([a], strategy)
        node_ids = {n.id for n in result}
        assert a.id in node_ids
        assert b.id in node_ids
        assert c.id not in node_ids  # depth 2 には到達しない


# ── ContextBuilder ────────────────────────────────────────────────────────────

class TestContextBuilder:
    def test_empty_nodes_returns_fallback(self):
        builder = ContextBuilder()
        ctx = builder.build("なぜ売上が落ちたか？", QueryType.WHY, [])
        assert "見つかりませんでした" in ctx.context_text

    def test_context_contains_node_text(self):
        nodes = [
            _make_node(NodeType.EVIDENCE, "在庫調整が主因。"),
            _make_node(NodeType.CLAIM, "売上が減少した。"),
        ]
        builder = ContextBuilder()
        ctx = builder.build("なぜ売上が落ちたか？", QueryType.WHY, nodes)
        assert "在庫調整が主因。" in ctx.context_text
        assert "売上が減少した。" in ctx.context_text

    def test_node_type_labels_appear(self):
        nodes = [_make_node(NodeType.EVIDENCE, "根拠テキスト。")]
        builder = ContextBuilder()
        ctx = builder.build("query", QueryType.WHY, nodes)
        assert "【根拠】" in ctx.context_text

    def test_max_nodes_limits_output(self):
        nodes = [_make_node(NodeType.CLAIM, f"主張{i}。") for i in range(10)]
        builder = ContextBuilder(max_nodes=3)
        ctx = builder.build("query", QueryType.WHAT, nodes)
        assert ctx.num_nodes == 3

    def test_why_puts_evidence_before_claim(self):
        """WHY型ではEVIDENCEがCLAIMより先に表示される。"""
        nodes = [
            _make_node(NodeType.CLAIM, "CLAIM_TEXT"),
            _make_node(NodeType.EVIDENCE, "EVIDENCE_TEXT"),
        ]
        builder = ContextBuilder()
        ctx = builder.build("why?", QueryType.WHY, nodes)
        evidence_pos = ctx.context_text.index("EVIDENCE_TEXT")
        claim_pos = ctx.context_text.index("CLAIM_TEXT")
        assert evidence_pos < claim_pos
