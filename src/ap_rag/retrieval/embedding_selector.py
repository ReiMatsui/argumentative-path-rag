"""
埋め込みベクトルベースの入口ノード選定モジュール。

研究計画書 §4.1 「② 入口選定 (E5-Mistral)」の実装。

BM25NodeSelector（キーワードマッチ）に代わり、dense embedding による
意味的類似度でクエリに最も関連する入口ノードを選定する。

**E5-Mistral-7B-Instruct の使い方 (公式推奨フォーマット)**
  - クエリ側: "Instruct: {task}\\nQuery: {query}"
  - 文書側  : テキストそのまま（プレフィックス不要）

**モデルサイズに関する注意**
  E5-Mistral-7B-Instruct は 7B パラメータ（fp16 で ~14GB）。
  MacBook 等のローカル環境では以下を推奨:
    - Apple Silicon (M1/M2/M3): EMBEDDING_DEVICE=mps  → 数秒/バッチ
    - GPU (CUDA):               EMBEDDING_DEVICE=cuda → 1秒以下/バッチ
    - CPU のみ:                 EMBEDDING_DEVICE=cpu  → 数分/バッチ (非推奨)

  開発・デバッグ時は軽量モデルで代替可:
    EMBEDDING_MODEL=intfloat/e5-large-v2          # 300M, 精度はやや落ちる
    EMBEDDING_MODEL=BAAI/bge-large-en-v1.5        # 335M, 研究計画の強いベースライン候補

**ノード埋め込みのキャッシュ戦略**
  ArgumentNode は frozen（不変）なため embedding を後付けできない。
  EmbeddingNodeSelector がインスタンス内の dict でキャッシュを管理し、
  同一ノードへの再計算を避ける。パイプラインのインスタンスが生きている間は
  キャッシュが保持される（QASPER 評価では論文10件分のノードが蓄積される）。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ap_rag.models.graph import ArgumentNode

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# E5-Mistral のクエリ側インストラクション
_E5_INSTRUCT = (
    "Retrieve the most relevant argumentative node "
    "that answers or supports the given question"
)


def _is_e5_mistral(model_name: str) -> bool:
    """E5-Mistral 系モデルかどうかを判定する（フォーマット切り替えに使用）。"""
    name_lower = model_name.lower()
    return "e5-mistral" in name_lower or "e5_mistral" in name_lower


class EmbeddingNodeSelector:
    """dense embedding によるクエリ関連度ランキングで入口ノードを選定する。

    初回呼び出し時にモデルを遅延ロードし、以降はキャッシュを使い回す。

    Args:
        model_name: sentence-transformers モデル名または HuggingFace モデル ID。
                    デフォルト "intfloat/e5-mistral-7b-instruct"。
        device:     "cpu" | "cuda" | "mps"。
        top_k:      返す上位ノード数。候補数がこれ以下なら全て返す。
        batch_size: 埋め込み計算のバッチサイズ。
                    大きいほど速いが VRAM/RAM を多く消費する。
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        device: str = "cpu",
        top_k: int = 10,
        batch_size: int = 32,
        encoder: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._top_k = top_k
        self._batch_size = batch_size
        self._is_e5_mistral = _is_e5_mistral(model_name)

        # 遅延ロード: 実際に使われるまでモデルをメモリに載せない
        # encoder が外部から提供された場合は遅延ロードをスキップする
        # （例: OpenAIEncoder を注入して sentence-transformers + torch 依存を回避）
        self._model: SentenceTransformer | None = encoder  # type: ignore[assignment]

        # node_id → 正規化済み embedding ベクトル のキャッシュ
        self._cache: dict[str, np.ndarray] = {}

    # ── 公開 API ──────────────────────────────────────────────────────────────

    def select(
        self,
        nodes: list[ArgumentNode],
        query: str,
    ) -> list[ArgumentNode]:
        """クエリに最も意味的に近い上位 top_k ノードを返す。

        フォールバック動作:
        - ノード数 ≤ top_k → 全て返す（埋め込み計算不要）
        - ノード数が 0    → 空リストを返す

        Args:
            nodes: 候補ノード（同一クエリ型の入口候補）。
            query: ユーザーのクエリ文字列。

        Returns:
            コサイン類似度降順の上位 top_k ノード。
        """
        if len(nodes) == 0:
            return []
        if len(nodes) <= self._top_k:
            return nodes

        model = self._get_model()

        # 未キャッシュのノードを一括エンコード
        self._encode_nodes(nodes, model)

        # クエリをエンコード
        query_text = self._format_query(query)
        query_emb = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        query_emb = np.array(query_emb, dtype=np.float32)

        # コサイン類似度 = 正規化済みベクトルの内積
        node_embs = np.stack([self._cache[n.id] for n in nodes])  # (N, D)
        scores: np.ndarray = node_embs @ query_emb                # (N,)

        top_indices = np.argsort(scores)[::-1][: self._top_k]
        selected = [nodes[int(i)] for i in top_indices]

        logger.info(
            "EmbeddingNodeSelector: 候補 %d → top %d 選定 (best_score=%.3f)",
            len(nodes),
            len(selected),
            float(scores[top_indices[0]]),
        )
        return selected

    def cache_size(self) -> int:
        """キャッシュ済みノード数を返す（デバッグ用）。"""
        return len(self._cache)

    def get_encoder(self) -> SentenceTransformer:
        """遅延ロード済みの SentenceTransformer を公開 API として返す。

        他コンポーネント（`CrossChunkEdgeExtractor` など）が同じモデルを
        共有して、E5-Mistral 等の巨大モデルを二重ロードしないために使う。
        """
        return self._get_model()

    # ── private ───────────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """モデルを遅延ロードして返す。"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers が必要です: pip install sentence-transformers"
                ) from e

            logger.info(
                "埋め込みモデルをロード中: %s (device=%s) ...",
                self._model_name,
                self._device,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
            logger.info("埋め込みモデルのロード完了")
        return self._model

    def _encode_nodes(
        self,
        nodes: list[ArgumentNode],
        model: SentenceTransformer,
    ) -> None:
        """キャッシュにないノードをバッチエンコードしてキャッシュに格納する。"""
        uncached = [n for n in nodes if n.id not in self._cache]
        if not uncached:
            return

        texts = [self._format_passage(n) for n in uncached]
        logger.debug("ノード埋め込み計算: %d 件", len(texts))

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        for node, emb in zip(uncached, embeddings):
            self._cache[node.id] = emb

    def _format_query(self, query: str) -> str:
        """モデルに合わせたクエリフォーマットを返す。

        E5-Mistral-7B-Instruct は "Instruct: {task}\\nQuery: {text}" 形式を要求する。
        他の E5 系・BGE 系はテキストそのまま（またはプレフィックス付き）。
        """
        if self._is_e5_mistral:
            return f"Instruct: {_E5_INSTRUCT}\nQuery: {query}"
        return query

    def _format_passage(self, node: ArgumentNode) -> str:
        """ノードテキストをモデルに合わせてフォーマットする。

        E5-Mistral はパッセージ側にプレフィックス不要。
        ノードタイプをプレフィックスとして付与することで、モデルが
        「これはCLAIMである」という文脈を持てるようにする。
        embedding ターゲットには原文 span を優先する (verbatim_text)。
        """
        text = getattr(node, "verbatim_text", None) or node.text
        return f"{node.node_type.value}: {text}"
