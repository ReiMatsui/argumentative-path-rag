"""
Argumentative-Path RAG のエンドツーエンドパイプライン。

研究計画書 §4.1「オンライン（クエリ時）」フローを実装する:
  クエリ → ①型分類 → ②入口選定 → ③型適応探索 → ④コンテキスト構築 → LLM回答

Usage:
    from ap_rag.pipeline import ArgumentativeRAGPipeline, PipelineFactory

    pipeline = PipelineFactory.from_settings(settings)
    result = pipeline.query("なぜQ3の売上が落ちたか？", doc_id="doc_001")
    print(result.answer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ap_rag.config import Settings
from ap_rag.generation.generator import AnswerGenerator, GenerationResult
from ap_rag.graph.neo4j_store import Neo4jGraphStore
from ap_rag.graph.store import GraphStore
from ap_rag.indexing.chunker import SentenceChunker
from ap_rag.indexing.classifier import NodeClassifier
from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.indexing.pipeline import IndexingPipeline, IndexingResult
from ap_rag.models.taxonomy import TRAVERSAL_STRATEGIES, NodeType, QueryType
from ap_rag.retrieval.context_builder import ContextBuilder, RetrievalContext
from ap_rag.retrieval.embedding_selector import EmbeddingNodeSelector
from ap_rag.retrieval.query_classifier import QueryClassifier
from ap_rag.retrieval.selector import BM25NodeSelector
from ap_rag.retrieval.traversal import GraphTraverser

logger = logging.getLogger(__name__)


class ArgumentativeRAGPipeline:
    """Argumentative-Path RAG のオンライン検索パイプライン。

    Args:
        store: 議論グラフのストア（Neo4j または NetworkX）。
        query_classifier: クエリ型分類器。
        traverser: グラフ探索器。
        context_builder: コンテキスト構築器。
        generator: 回答生成器。
        node_selector: BM25 ベースの入口ノード選定器（省略時は内部でデフォルト生成）。
    """

    def __init__(
        self,
        store: GraphStore,
        query_classifier: QueryClassifier,
        traverser: GraphTraverser,
        context_builder: ContextBuilder,
        generator: AnswerGenerator,
        node_selector: EmbeddingNodeSelector | BM25NodeSelector | None = None,
    ) -> None:
        self._store = store
        self._query_classifier = query_classifier
        self._traverser = traverser
        self._context_builder = context_builder
        self._generator = generator
        # デフォルトは BM25（軽量・依存なし）。
        # 研究計画通りに E5-Mistral を使う場合は PipelineFactory.from_settings() を利用。
        self._node_selector: EmbeddingNodeSelector | BM25NodeSelector = (
            node_selector if node_selector is not None else BM25NodeSelector(top_k=10)
        )

    def query(self, query: str, doc_id: str) -> GenerationResult:
        """クエリに対する回答を生成して返す。

        Args:
            query: ユーザーの質問。
            doc_id: 検索対象の文書ID。

        Returns:
            GenerationResult — 回答と取得されたコンテキストを含む。
        """
        # Step 1: クエリ型分類
        query_type = self._query_classifier.classify(query)
        logger.info("クエリ型: %s — %r", query_type.value, query)

        # Step 2: 入口ノードの選定（BM25 クエリ関連度フィルタ付き）
        entry_nodes = self._select_entry_nodes(doc_id, query_type, query)
        if not entry_nodes:
            logger.warning("入口ノードが見つかりません: doc_id=%s", doc_id)

        # Step 3: 型適応探索
        retrieved_nodes = self._traverser.traverse(entry_nodes, query_type)
        logger.info("取得ノード数: %d", len(retrieved_nodes))

        # Step 4: コンテキスト構築
        context = self._context_builder.build(query, query_type, retrieved_nodes)

        # Step 5: LLMで回答
        result = self._generator.generate(context)
        return result

    # ── private ────────────────────────────────────────────────────────────

    def _select_entry_nodes(
        self,
        doc_id: str,
        query_type: QueryType,
        query: str,
    ):
        """探索戦略に基づいて入口ノードを取得し、BM25 でクエリ関連度フィルタをかける。

        旧実装は「型に合致する全ノードを返す」だけで、350 ノードの論文では
        150 以上の無関係な CLAIM が候補になっていた（クエリ盲目問題）。

        改善 (v2):
          1. 型で候補を絞る（従来通り）
          2. BM25 でクエリとの関連度を計算し上位 top_k のみ返す
          → BFS 探索の出発点が質問に関連したノードになり、
            コンテキストの質が大幅に向上する。
        """
        strategy = TRAVERSAL_STRATEGIES[query_type]
        candidate_nodes: list = []
        for node_type in strategy.entry_node_types:
            nodes = self._store.get_nodes_by_type(doc_id, node_type)
            candidate_nodes.extend(nodes)

        # BM25 でクエリ関連度の高い入口ノードのみに絞る
        entry_nodes = self._node_selector.select(candidate_nodes, query)
        logger.info(
            "入口ノード: 候補 %d → BM25 選定後 %d (query_type=%s)",
            len(candidate_nodes),
            len(entry_nodes),
            query_type.value,
        )
        return entry_nodes


class PipelineFactory:
    """設定から ArgumentativeRAGPipeline を構築するファクトリクラス。"""

    @staticmethod
    def from_settings(settings: Settings) -> ArgumentativeRAGPipeline:
        """Settings オブジェクトからパイプラインを構築する。"""
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai パッケージが必要です: pip install openai") from e

        client = openai.OpenAI(api_key=settings.openai_api_key)
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        node_selector = EmbeddingNodeSelector(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            top_k=10,
        )
        return ArgumentativeRAGPipeline(
            store=store,
            query_classifier=QueryClassifier(
                client=client,
                model=settings.openai_generator_model,
            ),
            traverser=GraphTraverser(store=store),
            context_builder=ContextBuilder(),
            generator=AnswerGenerator(
                client=client,
                model=settings.openai_generator_model,
            ),
            node_selector=node_selector,
        )

    @staticmethod
    def build_indexing_pipeline(settings: Settings) -> IndexingPipeline:
        """インデックスパイプラインを構築する。"""
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai パッケージが必要です") from e

        client = openai.OpenAI(api_key=settings.openai_api_key)
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        return IndexingPipeline(
            chunker=SentenceChunker(),
            classifier=NodeClassifier(
                client=client,
                model=settings.openai_classifier_model,
            ),
            extractor=EdgeExtractor(
                client=client,
                model=settings.openai_classifier_model,
            ),
            store=store,
        )
