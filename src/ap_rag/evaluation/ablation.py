"""
アブレーション実験基盤。

研究計画書 §4.4「主要なアブレーション」:
  ★★★ LLMリライト＋標準RAG との比較 — 改善がグラフ構造由来か LLM前処理由来かを分離
  ★★  探索戦略を固定化          — クエリ型適応（新規貢献①）の寄与を測定
  ★★  グラフ構築誤差耐性         — 分類精度を人工劣化し、臨界点を把握
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ap_rag.evaluation.baselines import BaselineResult, _TextNode, BM25RAG
from ap_rag.models.graph import ArgumentNode
from ap_rag.models.taxonomy import NodeType, QueryType, TRAVERSAL_STRATEGIES
from ap_rag.openai_compat import max_tokens_kwarg, reasoning_kwarg, sampling_kwargs

logger = logging.getLogger(__name__)


# ── LLMリライト＋標準RAG ──────────────────────────────────────────────────────

class LLMRewriteRAG:
    """アブレーション用: LLMでチャンクをリライトしてから BM25 で検索する。

    これを ArgumentativeRAG と比較することで、
    「改善はグラフ構造由来か、LLM前処理由来か」を切り分けられる。

    研究計画書 §4.4 ★★★
    """

    _REWRITE_PROMPT = """\
Rewrite the following text to make key claims and evidence more explicit and retrievable. \
Preserve all factual information. Output only the rewritten text, no explanations.

Text: {text}
"""

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._bm25 = BM25RAG(client=client, model=model, top_k=top_k)

    def index(self, texts: list[str], doc_id: str) -> None:
        """チャンクをLLMでリライトしてからBM25でインデックスする。"""
        rewritten = [self._rewrite(t) for t in texts]
        self._bm25.index(rewritten, doc_id)

    def query(self, question: str, doc_id: str) -> BaselineResult:
        return self._bm25.query(question, doc_id)

    def _rewrite(self, text: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{
                    "role": "user",
                    "content": self._REWRITE_PROMPT.format(text=text),
                }],
                **sampling_kwargs(self._model, temperature=0.1),
                **max_tokens_kwarg(self._model, 300),
                **reasoning_kwarg(self._model),
            )
            return (response.choices[0].message.content or text).strip()
        except Exception as e:
            logger.warning("リライト失敗: %s", e)
            return text


# ── 探索戦略固定化 ────────────────────────────────────────────────────────────

@dataclass
class FixedStrategyResult:
    """固定探索戦略版パイプラインの結果。"""
    answer: str
    retrieval_context: Any


class FixedStrategyRAG:
    """アブレーション用: クエリ型に関係なく常に同一の探索戦略を使うRAG。

    クエリ型適応（新規貢献①）の寄与を測定する。
    研究計画書 §4.4 ★★
    """

    def __init__(
        self,
        store: Any,
        generator: Any,
        context_builder: Any,
        fixed_query_type: QueryType = QueryType.WHY,
    ) -> None:
        self._store = store
        self._generator = generator
        self._context_builder = context_builder
        self._fixed_query_type = fixed_query_type
        from ap_rag.retrieval.traversal import GraphTraverser
        self._traverser = GraphTraverser(store)

    def query(self, question: str, doc_id: str) -> FixedStrategyResult:
        """常に fixed_query_type の探索戦略を使って回答する。"""
        strategy = TRAVERSAL_STRATEGIES[self._fixed_query_type]
        entry_nodes: list[ArgumentNode] = []
        for nt in strategy.entry_node_types:
            entry_nodes.extend(self._store.get_nodes_by_type(doc_id, nt))

        retrieved = self._traverser.traverse(entry_nodes, self._fixed_query_type)
        context = self._context_builder.build(question, self._fixed_query_type, retrieved)
        result = self._generator.generate(context)

        return FixedStrategyResult(answer=result.answer, retrieval_context=context)


# ── グラフ構築誤差耐性 ────────────────────────────────────────────────────────

class NoisyGraphStore:
    """アブレーション用: ノード型をランダムに誤分類したノイズ入りグラフストア。

    実際のグラフストアをラップし、get_nodes_by_type と get_neighbors の
    返値を error_rate の確率でランダムなノード型に置き換える。

    研究計画書 §4.4 ★★
    """

    def __init__(self, base_store: Any, error_rate: float = 0.1, seed: int = 42) -> None:
        self._store = base_store
        self._error_rate = error_rate
        self._rng = random.Random(seed)
        self._all_types = list(NodeType)

    def get_nodes_by_type(self, doc_id: str, node_type: NodeType) -> list[ArgumentNode]:
        nodes = self._store.get_nodes_by_type(doc_id, node_type)
        return [self._maybe_corrupt(n) for n in nodes]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "incoming",
    ) -> list[ArgumentNode]:
        nodes = self._store.get_neighbors(node_id, edge_types=edge_types, direction=direction)
        return [self._maybe_corrupt(n) for n in nodes]

    def _maybe_corrupt(self, node: ArgumentNode) -> ArgumentNode:
        """error_rate の確率でノード型をランダムに変える。"""
        if self._rng.random() < self._error_rate:
            wrong_types = [t for t in self._all_types if t != node.node_type]
            wrong_type = self._rng.choice(wrong_types)
            # frozen=True なので model_copy で新インスタンスを作る
            return node.model_copy(update={"node_type": wrong_type})
        return node

    # 残りのメソッドは base_store に委譲
    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


# ── アブレーション実験スイート ────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """アブレーション実験の設定。"""
    name: str
    description: str
    error_rates: list[float] = None  # NoisyGraph 用

    def __post_init__(self):
        if self.error_rates is None:
            self.error_rates = [0.0, 0.1, 0.2, 0.3, 0.5]


ABLATION_CONFIGS = [
    AblationConfig(
        name="llm_rewrite_bm25",
        description="LLMリライト＋BM25（グラフ構造の寄与を分離）",
    ),
    AblationConfig(
        name="fixed_strategy_why",
        description="全クエリをWHY型固定で探索（クエリ型適応の寄与を測定）",
    ),
    AblationConfig(
        name="noisy_graph",
        description="グラフ構築誤差耐性（error_rate を段階的に増加）",
        error_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
    ),
]
