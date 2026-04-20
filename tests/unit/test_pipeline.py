"""エンドツーエンドパイプラインのユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock

from ap_rag.generation.generator import AnswerGenerator, GenerationResult
from ap_rag.graph.store import GraphStore
from ap_rag.models import ArgumentNode, NodeType, QueryType
from ap_rag.pipeline import ArgumentativeRAGPipeline
from ap_rag.retrieval.context_builder import ContextBuilder
from ap_rag.retrieval.query_classifier import QueryClassifier
from ap_rag.retrieval.traversal import GraphTraverser


def _make_node(node_type: NodeType, text: str) -> ArgumentNode:
    return ArgumentNode(
        node_type=node_type,
        text=text,
        source_doc_id="doc_001",
        source_chunk_idx=0,
    )


def _build_pipeline(
    query_type: QueryType = QueryType.WHY,
    store_nodes: list[ArgumentNode] | None = None,
    traversal_nodes: list[ArgumentNode] | None = None,
    answer: str = "これがテスト回答です。",
) -> ArgumentativeRAGPipeline:
    """テスト用パイプラインを構築するヘルパー。"""
    store_nodes = store_nodes or []
    traversal_nodes = traversal_nodes or []

    # QueryClassifier モック
    mock_classifier = MagicMock(spec=QueryClassifier)
    mock_classifier.classify.return_value = query_type

    # GraphStore モック
    mock_store = MagicMock(spec=GraphStore)
    mock_store.get_nodes_by_type.return_value = store_nodes

    # GraphTraverser モック
    mock_traverser = MagicMock(spec=GraphTraverser)
    mock_traverser.traverse.return_value = traversal_nodes

    # AnswerGenerator モック
    mock_generator = MagicMock(spec=AnswerGenerator)
    mock_generator.generate.return_value = GenerationResult(
        query="test query",
        answer=answer,
        retrieval_context=ContextBuilder().build("test query", query_type, traversal_nodes),
        model="gpt-4o-mini",
    )

    return ArgumentativeRAGPipeline(
        store=mock_store,
        query_classifier=mock_classifier,
        traverser=mock_traverser,
        context_builder=ContextBuilder(),
        generator=mock_generator,
    )


class TestArgumentativeRAGPipeline:
    def test_query_returns_answer(self):
        pipeline = _build_pipeline(answer="Q3の売上減少は在庫調整が原因です。")
        result = pipeline.query("なぜQ3の売上が落ちたか？", doc_id="doc_001")
        assert result.answer == "Q3の売上減少は在庫調整が原因です。"

    def test_query_type_is_classified(self):
        pipeline = _build_pipeline(query_type=QueryType.WHY)
        result = pipeline.query("なぜQ3が落ちたか？", doc_id="doc_001")
        assert result.retrieval_context.query_type == QueryType.WHY

    def test_traversal_is_called_with_entry_nodes(self):
        entry_node = _make_node(NodeType.CLAIM, "Q3売上減少。")
        mock_classifier = MagicMock(spec=QueryClassifier)
        mock_classifier.classify.return_value = QueryType.WHY

        mock_store = MagicMock(spec=GraphStore)
        mock_store.get_nodes_by_type.return_value = [entry_node]

        mock_traverser = MagicMock(spec=GraphTraverser)
        mock_traverser.traverse.return_value = [entry_node]

        mock_generator = MagicMock(spec=AnswerGenerator)
        mock_generator.generate.return_value = GenerationResult(
            query="q",
            answer="a",
            retrieval_context=ContextBuilder().build("q", QueryType.WHY, [entry_node]),
            model="m",
        )

        pipeline = ArgumentativeRAGPipeline(
            store=mock_store,
            query_classifier=mock_classifier,
            traverser=mock_traverser,
            context_builder=ContextBuilder(),
            generator=mock_generator,
        )
        pipeline.query("なぜ？", doc_id="doc_001")

        # traverser が entry_nodes を受け取って呼ばれたことを検証
        mock_traverser.traverse.assert_called_once()
        call_args = mock_traverser.traverse.call_args
        assert entry_node in call_args[0][0]

    def test_empty_entry_nodes_still_returns_result(self):
        """入口ノードがなくても例外にならず、空コンテキストで回答を返す。"""
        pipeline = _build_pipeline(
            store_nodes=[],
            traversal_nodes=[],
            answer="情報が不足しています。",
        )
        result = pipeline.query("なぜ？", doc_id="doc_001")
        assert result.answer == "情報が不足しています。"
