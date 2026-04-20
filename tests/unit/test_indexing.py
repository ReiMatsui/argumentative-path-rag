"""インデックスパイプライン各コンポーネントのユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ap_rag.indexing.chunker import DocumentChunk, SentenceChunker
from ap_rag.indexing.classifier import NodeClassifier
from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.indexing.pipeline import IndexingPipeline
from ap_rag.indexing.schemas import (
    EdgeExtractionOutput,
    EdgeItem,
    NodeClassificationOutput,
    NodeItem,
)
from ap_rag.graph.store import GraphStore
from ap_rag.models import ArgumentGraph, ArgumentNode, EdgeType, NodeType


# ── モックヘルパー ─────────────────────────────────────────────────────────────

def _make_parse_client(parsed_output) -> MagicMock:
    """client.beta.chat.completions.parse() を返すモッククライアント。

    parsed_output には NodeClassificationOutput / EdgeExtractionOutput を渡す。
    None を渡すと refusal（parsed=None）のシミュレーションになる。
    """
    mock_message = MagicMock()
    mock_message.parsed = parsed_output
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response
    return mock_client


# ── SentenceChunker ───────────────────────────────────────────────────────────

class TestSentenceChunker:
    def test_single_sentence(self):
        chunker = SentenceChunker(max_tokens=512)
        chunks = chunker.chunk("Q3売上は12%減少した。", doc_id="doc_001")
        assert len(chunks) >= 1
        assert chunks[0].doc_id == "doc_001"
        assert chunks[0].chunk_idx == 0

    def test_multiple_sentences(self):
        text = "Q3売上は12%減少した。競合X社は8%増加した。電子部品在庫調整が主因だ。"
        chunker = SentenceChunker(max_tokens=512)
        chunks = chunker.chunk(text, doc_id="doc_001")
        assert len(chunks) >= 1
        combined = " ".join(c.text for c in chunks)
        assert "Q3売上" in combined

    def test_chunk_idx_is_sequential(self):
        text = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        chunker = SentenceChunker(max_tokens=30)
        chunks = chunker.chunk(text, doc_id="doc_x")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_idx == i


# ── NodeClassifier ────────────────────────────────────────────────────────────

class TestNodeClassifier:
    def _make_chunk(self) -> DocumentChunk:
        return DocumentChunk(
            doc_id="doc_001",
            chunk_idx=0,
            text="Q3売上は12%減少した。電子部品の在庫調整が主因である。",
            char_start=0,
            char_end=50,
        )

    def test_normal_classification(self):
        """正常系: 2ノードが返る。"""
        output = NodeClassificationOutput(nodes=[
            NodeItem(node_type=NodeType.CLAIM,    text="Q3売上は12%減少した。"),
            NodeItem(node_type=NodeType.EVIDENCE, text="電子部品の在庫調整が主因である。"),
        ])
        client = _make_parse_client(output)
        classifier = NodeClassifier(client=client, model="gpt-4o")
        nodes = classifier.classify(self._make_chunk())
        assert len(nodes) == 2
        assert nodes[0].node_type == NodeType.CLAIM
        assert nodes[1].node_type == NodeType.EVIDENCE

    def test_empty_nodes_returns_empty(self):
        """nodes が空の出力は空リストを返す。"""
        output = NodeClassificationOutput(nodes=[])
        client = _make_parse_client(output)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert nodes == []

    def test_refusal_returns_empty(self):
        """API が refusal を返した場合（parsed=None）は空リストになる。"""
        client = _make_parse_client(None)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert nodes == []

    def test_blank_text_node_is_skipped(self):
        """text が空白のノードはスキップされる。"""
        output = NodeClassificationOutput(nodes=[
            NodeItem(node_type=NodeType.CLAIM, text="   "),
            NodeItem(node_type=NodeType.EVIDENCE, text="有効なノード。"),
        ])
        client = _make_parse_client(output)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert len(nodes) == 1
        assert nodes[0].node_type == NodeType.EVIDENCE

    def test_invalid_node_type_rejected_by_pydantic(self):
        """NodeType 列挙値にない文字列は Pydantic がバリデーションエラーを出す。"""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            NodeItem(node_type="QUESTION", text="テスト")

    def test_node_has_correct_doc_id(self):
        """生成されたノードに chunk の doc_id が設定される。"""
        output = NodeClassificationOutput(nodes=[
            NodeItem(node_type=NodeType.CLAIM, text="主張テキスト。"),
        ])
        client = _make_parse_client(output)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert nodes[0].source_doc_id == "doc_001"
        assert nodes[0].source_chunk_idx == 0


# ── EdgeExtractor ─────────────────────────────────────────────────────────────

class TestEdgeExtractor:
    def _make_nodes(self) -> list[ArgumentNode]:
        return [
            ArgumentNode(
                node_type=NodeType.EVIDENCE,
                text="電子部品の在庫調整が主因。",
                source_doc_id="doc_001",
                source_chunk_idx=0,
            ),
            ArgumentNode(
                node_type=NodeType.CLAIM,
                text="Q3売上は12%減少した。",
                source_doc_id="doc_001",
                source_chunk_idx=0,
            ),
        ]

    def test_single_node_returns_empty(self):
        """ノードが1件以下の場合はLLMを呼ばず空リストを返す。"""
        extractor = EdgeExtractor(client=MagicMock())
        assert extractor.extract(self._make_nodes()[:1]) == []

    def test_normal_extraction(self):
        """正常系: SUPPORTS エッジが1件返る。"""
        output = EdgeExtractionOutput(edges=[
            EdgeItem(source_idx=0, target_idx=1, edge_type=EdgeType.SUPPORTS, confidence=0.9),
        ])
        client = _make_parse_client(output)
        nodes = self._make_nodes()
        extractor = EdgeExtractor(client=client, min_confidence=0.7)
        edges = extractor.extract(nodes)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SUPPORTS
        assert edges[0].source_id == nodes[0].id
        assert edges[0].target_id == nodes[1].id

    def test_low_confidence_edge_is_filtered(self):
        """min_confidence 未満のエッジは除外される。"""
        output = EdgeExtractionOutput(edges=[
            EdgeItem(source_idx=0, target_idx=1, edge_type=EdgeType.SUPPORTS, confidence=0.5),
        ])
        client = _make_parse_client(output)
        extractor = EdgeExtractor(client=client, min_confidence=0.7)
        edges = extractor.extract(self._make_nodes())
        assert edges == []

    def test_self_loop_is_filtered(self):
        """自己ループ（source_idx == target_idx）は除外される。"""
        output = EdgeExtractionOutput(edges=[
            EdgeItem(source_idx=0, target_idx=0, edge_type=EdgeType.SUPPORTS, confidence=0.95),
        ])
        client = _make_parse_client(output)
        extractor = EdgeExtractor(client=client)
        edges = extractor.extract(self._make_nodes())
        assert edges == []

    def test_out_of_range_index_is_filtered(self):
        """ノード数を超えたインデックスは除外される。"""
        output = EdgeExtractionOutput(edges=[
            EdgeItem(source_idx=0, target_idx=99, edge_type=EdgeType.SUPPORTS, confidence=0.9),
        ])
        client = _make_parse_client(output)
        extractor = EdgeExtractor(client=client)
        edges = extractor.extract(self._make_nodes())
        assert edges == []

    def test_refusal_returns_empty(self):
        """API が refusal を返した場合（parsed=None）は空リストになる。"""
        client = _make_parse_client(None)
        extractor = EdgeExtractor(client=client)
        edges = extractor.extract(self._make_nodes())
        assert edges == []

    def test_invalid_edge_type_rejected_by_pydantic(self):
        """EdgeType 列挙値にない文字列は Pydantic が拒否する。"""
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            EdgeItem(source_idx=0, target_idx=1, edge_type="UNKNOWN")


# ── IndexingPipeline ──────────────────────────────────────────────────────────

class TestIndexingPipeline:
    def _make_pipeline(self, classifier_nodes, extractor_edges) -> IndexingPipeline:
        chunker = SentenceChunker(max_tokens=512)

        mock_classifier = MagicMock(spec=NodeClassifier)
        mock_classifier.classify.return_value = classifier_nodes

        mock_extractor = MagicMock(spec=EdgeExtractor)
        mock_extractor.extract.return_value = extractor_edges

        mock_store = MagicMock(spec=GraphStore)

        return IndexingPipeline(
            chunker=chunker,
            classifier=mock_classifier,
            extractor=mock_extractor,
            store=mock_store,
            show_progress=False,
        )

    def test_end_to_end_run(self):
        node_ev = ArgumentNode(
            node_type=NodeType.EVIDENCE,
            text="電子部品の在庫調整が主因。",
            source_doc_id="doc_001",
            source_chunk_idx=0,
        )
        node_cl = ArgumentNode(
            node_type=NodeType.CLAIM,
            text="Q3売上は12%減少した。",
            source_doc_id="doc_001",
            source_chunk_idx=0,
        )
        pipeline = self._make_pipeline(
            classifier_nodes=[node_ev, node_cl],
            extractor_edges=[],
        )
        result = pipeline.run(
            text="電子部品の在庫調整が主因。Q3売上は12%減少した。",
            doc_id="doc_001",
        )
        assert result.num_nodes == 2
        assert result.doc_id == "doc_001"
