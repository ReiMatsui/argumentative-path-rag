"""
チャンク境界をまたぐエッジ抽出モジュール。

旧実装は `EdgeExtractor` が 1 チャンク内のノードだけを見ていたため、
チャンク間の関連（遠くの EVIDENCE が CLAIM を支持する等）が失われていた。

本モジュールは以下の流れでチャンク間エッジを補う:

  1. グラフ中の全ノードの埋め込みベクトルを計算
  2. 各ノードについて、埋め込み類似度で上位 K 件の近傍を抽出
     （ただし同一チャンクのノードと既に直接辺のあるノードは除外）
  3. 近傍ペアの両端を含む小クラスタを構成
  4. 既存の `EdgeExtractor` にクラスタを渡して LLM に辺を判定させる

方針:
  * LLM の全数判定 (O(N²)) は長論文で爆発するので、埋め込みで候補を絞る
  * 同一チャンク内は EdgeExtractor で既に処理済みなので除外
  * 既存の辺も除外することで LLM コストを抑える
  * type-incompatible なペア（例: CLAIM-CLAIM のみ）でも LLM が
    「edges=[]」を返せば済むので、型プレフィルタは粗めで十分
"""

from __future__ import annotations

import logging
from typing import Any

from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.models.graph import ArgumentEdge, ArgumentNode

logger = logging.getLogger(__name__)


class CrossChunkEdgeExtractor:
    """チャンク間エッジの補完器。

    Args:
        edge_extractor: LLM ベースの辺抽出器（既存の `EdgeExtractor`）。
        encoder: 埋め込み関数を持つオブジェクト（`.encode(texts: list[str]) -> np.ndarray`）。
                 None の場合は sentence-transformers を遅延ロードする。
        embedding_model: encoder が None のときに使う SentenceTransformer モデル名。
        device: 埋め込みモデルのデバイス。
        top_k: 各ノードについて探す近傍数。
        batch_size: LLM に渡す1バッチあたりのノード数（小さくすると精度↑、コスト↑）。
        min_similarity: この類似度未満の近傍は候補から除外する。
    """

    def __init__(
        self,
        edge_extractor: EdgeExtractor,
        encoder: Any | None = None,
        embedding_model: str = "intfloat/e5-mistral-7b-instruct",
        device: str = "cpu",
        top_k: int = 5,
        batch_size: int = 8,
        min_similarity: float = 0.45,
    ) -> None:
        self._edge_extractor = edge_extractor
        self._encoder = encoder
        self._embedding_model = embedding_model
        self._device = device
        self._top_k = top_k
        self._batch_size = max(2, batch_size)
        self._min_similarity = min_similarity

    # ── public ─────────────────────────────────────────────────────────────

    def extract(
        self,
        nodes: list[ArgumentNode],
        existing_edges: list[ArgumentEdge],
    ) -> list[ArgumentEdge]:
        """チャンク間エッジ候補を返す。

        Args:
            nodes: 文書に属する全ノード。
            existing_edges: 既にグラフに追加されているエッジ（重複判定に使用）。

        Returns:
            LLM が認めた新規 `ArgumentEdge` のリスト。
        """
        if len(nodes) < 2:
            return []

        # 既存辺の (src_id, tgt_id) セット（無向で扱う）
        existing_pairs: set[frozenset[str]] = {
            frozenset({e.source_id, e.target_id}) for e in existing_edges
        }

        # 埋め込みを計算
        embeddings = self._embed([n.text for n in nodes])
        if embeddings is None:
            logger.warning(
                "CrossChunkEdgeExtractor: 埋め込みが取得できず、チャンク間抽出をスキップ"
            )
            return []

        # 候補ペア（同一チャンクでなく、既存辺でもないもの）を抽出
        candidate_pairs = self._build_candidate_pairs(
            nodes, embeddings, existing_pairs
        )
        if not candidate_pairs:
            logger.info("CrossChunkEdgeExtractor: 候補ペア 0 件")
            return []

        logger.info(
            "CrossChunkEdgeExtractor: ノード %d, 候補ペア %d",
            len(nodes),
            len(candidate_pairs),
        )

        # バッチ構成: 類似ペアが同じバッチにまとまるよう詰め直す
        batches = self._pack_indices(nodes, candidate_pairs)

        # LLM でバッチごとに判定
        new_edges: list[ArgumentEdge] = []
        seen_new: set[tuple[str, str, str]] = set()  # (src, tgt, edge_type)
        for batch in batches:
            try:
                edges = self._edge_extractor.extract(batch)
            except Exception as e:
                logger.warning("CrossChunkEdgeExtractor バッチ失敗: %s", e)
                continue

            for edge in edges:
                # 既存辺および同一バッチ内で既に返された辺を除外
                pair = frozenset({edge.source_id, edge.target_id})
                if pair in existing_pairs:
                    continue
                key = (edge.source_id, edge.target_id, edge.edge_type.value)
                if key in seen_new:
                    continue
                seen_new.add(key)
                new_edges.append(edge)

        logger.info("CrossChunkEdgeExtractor: 新規エッジ %d 件", len(new_edges))
        return new_edges

    # ── private ────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]):
        """ノードテキストを埋め込みベクトルにする。失敗時は None を返す。"""
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy が見つかりません")
            return None

        try:
            encoder = self._get_encoder()
            vecs = encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as e:
            logger.warning("埋め込み失敗: %s", e)
            return None

        return np.asarray(vecs)

    def _get_encoder(self) -> Any:
        """SentenceTransformer を遅延ロードする。"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self._embedding_model, device=self._device
            )
        return self._encoder

    def _build_candidate_pairs(
        self,
        nodes: list[ArgumentNode],
        embeddings,
        existing_pairs: set[frozenset[str]],
    ) -> list[tuple[int, int, float]]:
        """チャンクをまたぐ類似ペア候補のリストを返す。

        戻り値は (node_idx_a, node_idx_b, similarity) のタプル。
        """
        import numpy as np

        n = len(nodes)
        # 正規化済み → 内積がコサイン類似度
        sim = embeddings @ embeddings.T
        # 自分自身は -inf にして top-k で除外
        np.fill_diagonal(sim, -np.inf)

        candidate_set: set[tuple[int, int]] = set()
        for i in range(n):
            # 類似度降順の上位 top_k
            order = np.argsort(-sim[i])
            kept = 0
            for j in order:
                if kept >= self._top_k:
                    break
                if i == j:
                    continue
                score = float(sim[i, j])
                if score < self._min_similarity:
                    break  # 降順なので以降も全部下回る

                n_i, n_j = nodes[i], nodes[j]

                # 同一チャンクは EdgeExtractor で既に処理済み
                if (
                    n_i.source_doc_id == n_j.source_doc_id
                    and n_i.source_chunk_idx == n_j.source_chunk_idx
                ):
                    continue

                pair = frozenset({n_i.id, n_j.id})
                if pair in existing_pairs:
                    continue

                key = (min(i, j), max(i, j))
                if key in candidate_set:
                    kept += 1
                    continue
                candidate_set.add(key)
                kept += 1

        # タプルに変換（重複削除済み）
        pairs_with_score: list[tuple[int, int, float]] = []
        for a, b in candidate_set:
            pairs_with_score.append((a, b, float(sim[a, b])))
        # 類似度の高い順に並べる（重要ペアから消化される）
        pairs_with_score.sort(key=lambda t: -t[2])
        return pairs_with_score

    def _pack_indices(
        self,
        nodes: list[ArgumentNode],
        candidate_pairs: list[tuple[int, int, float]],
    ) -> list[list[ArgumentNode]]:
        """候補ペアをバッチに詰める（インデックスベース）。

        類似度の高いペアから処理し、そのペアが含まれる既存バッチを探す。
        見つからなければ新規バッチを作る。バッチサイズ上限は self._batch_size。
        """
        batches: list[set[int]] = []
        for a, b, _score in candidate_pairs:
            placed = False
            # 両方とも入る or 片方だけ入れれば他方と一緒になれるバッチを探す
            for bucket in batches:
                if a in bucket and b in bucket:
                    placed = True
                    break
                if len(bucket) + (0 if a in bucket else 1) + (0 if b in bucket else 1) \
                   <= self._batch_size:
                    if a in bucket or b in bucket:
                        bucket.add(a)
                        bucket.add(b)
                        placed = True
                        break
            if not placed:
                batches.append({a, b})

        # インデックス集合 → ノードリストに変換
        return [[nodes[i] for i in sorted(bucket)] for bucket in batches]
