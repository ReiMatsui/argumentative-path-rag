"""インデックスパイプライン各コンポーネントのユニットテスト。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ap_rag.indexing.chunker import DocumentChunk, SentenceChunker
from ap_rag.indexing.classifier import NodeClassifier
from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.indexing.pipeline import IndexingPipeline
from ap_rag.graph.store import GraphStore
from ap_rag.models import ArgumentGraph, ArgumentNode, EdgeType, NodeType


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
        # 全文が1チャンクにまとめられるか複数に分割されるかは max_tokens 次第
        assert len(chunks) >= 1
        # 全テキストが失われていないことを確認
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

    def _make_mock_client(self, response_json: str) -> MagicMock:
        """OpenAI クライアントのモックを生成する。"""
        mock_choice = MagicMock()
        mock_choice.message.content = response_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_normal_classification(self):
        response = json.dumps([
            {"node_type": "CLAIM", "text": "Q3売上は12%減少した。"},
            {"node_type": "EVIDENCE", "text": "電子部品の在庫調整が主因である。"},
        ])
        client = self._make_mock_client(response)
        classifier = NodeClassifier(client=client, model="gpt-4o")
        nodes = classifier.classify(self._make_chunk())
        assert len(nodes) == 2
        assert nodes[0].node_type == NodeType.CLAIM
        assert nodes[1].node_type == NodeType.EVIDENCE

    def test_dict_wrapper_response(self):
        """LLM が {"nodes": [...]} 形式で返す場合のテスト。"""
        response = json.dumps({"nodes": [
            {"node_type": "CLAIM", "text": "売上が減少した。"},
        ]})
        client = self._make_mock_client(response)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert len(nodes) == 1

    def test_invalid_node_type_is_skipped(self):
        response = json.dumps([
            {"node_type": "UNKNOWN_TYPE", "text": "何かテキスト。"},
            {"node_type": "CLAIM", "text": "有効なノード。"},
        ])
        client = self._make_mock_client(response)
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        # UNKNOWN_TYPE はスキップされ CLAIM だけ返る
        assert len(nodes) == 1
        assert nodes[0].node_type == NodeType.CLAIM

    def test_malformed_json_returns_empty(self):
        client = self._make_mock_client("これはJSONではない")
        classifier = NodeClassifier(client=client)
        nodes = classifier.classify(self._make_chunk())
        assert nodes == []


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

    def _make_mock_client(self, response_json: str) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = response_json
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_single_node_returns_empty(self):
        extractor = EdgeExtractor(client=MagicMock())
        nodes = self._make_nodes()[:1]
        assert extractor.extract(nodes) == []

    def test_normal_extraction(self):
        nodes = self._make_nodes()
        response = json.dumps([
            {"source_idx": 0, "target_idx": 1, "edge_type": "SUPPORTS", "confidence": 0.9}
        ])
        client = self._make_mock_client(response)
        extractor = EdgeExtractor(client=client, min_confidence=0.7)
        edges = extractor.extract(nodes)
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.SUPPORTS
        assert edges[0].source_id == nodes[0].id
        assert edges[0].target_id == nodes[1].id

    def test_low_confidence_edge_is_filtered(self):
        nodes = self._make_nodes()
        response = json.dumps([
            {"source_idx": 0, "target_idx": 1, "edge_type": "SUPPORTS", "confidence": 0.5}
        ])
        client = self._make_mock_client(response)
        extractor = EdgeExtractor(client=client, min_confidence=0.7)
        edges = extractor.extract(nodes)
        assert edges == []

    def test_self_loop_is_filtered(self):
        nodes = self._make_nodes()
        response = json.dumps([
            {"source_idx": 0, "target_idx": 0, "edge_type": "SUPPORTS", "confidence": 0.95}
        ])
        client = self._make_mock_client(response)
        extractor = EdgeExtractor(client=client)
        edges = extractor.extract(nodes)
        assert edges == []


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
