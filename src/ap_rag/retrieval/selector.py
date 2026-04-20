"""
BM25 ベースの入口ノード選定モジュール。

**問題の背景**:
旧実装の _select_entry_nodes は「指定型のノードを全て返す」だけだった。
350 ノードの論文では CLAIM だけで 150 個あり、ContextBuilder が上位 15 を
切り取っても、それはクエリと無関係な 15 ノードになる。BM25 ではなく
ランダムに近い選択になっており、BM25 ベースラインに大幅に劣る原因の一つだった。

**解決策**:
ノードテキストをコーパスとした BM25 でクエリとの関連度を計算し、
上位 top_k ノードだけを入口として返す。
グラフ探索で周辺ノードが補完されるため、入口の数は少なくてよい（デフォルト 10）。

**BM25 の選択理由**:
- 外部依存ゼロ（標準ライブラリのみ）
- 埋め込みモデル不要（インデックス時の embedding が不要なままにできる）
- TF-IDF より長い文書に対して公平（文書長正規化パラメータ b）
- キーワードマッチで "features", "dataset" などの具体語に強い
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ap_rag.models.graph import ArgumentNode


# ── トークナイザ ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """英語＋CJK 対応のシンプルなトークナイザ。

    英語: 小文字化してアルファベット/数字の連続を抽出。
    CJK: Unicode の基本漢字・ひらがな・カタカナ・ハングルを1文字ずつ。
    ストップワードや形態素解析は行わない（依存なし優先）。
    """
    text = text.lower()
    tokens = re.findall(
        r"[a-z0-9]+|[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\uac00-\ud7af]",
        text,
    )
    return tokens


# ── データクラス ────────────────────────────────────────────────────────────────

@dataclass
class _ScoredNode:
    node: ArgumentNode
    score: float


# ── BM25 セレクタ ───────────────────────────────────────────────────────────────

class BM25NodeSelector:
    """BM25 スコアでノードをクエリ関連度順にランク付けし、上位を返す。

    Args:
        k1:    用語頻度の飽和パラメータ（Robertson & Walker 推奨: 1.2〜2.0）。
        b:     文書長正規化パラメータ（0=正規化なし, 1=完全正規化, 通常 0.75）。
        top_k: 返す上位ノード数。ノード数がこれ以下なら全て返す。
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 10,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._top_k = top_k

    def select(
        self,
        nodes: list[ArgumentNode],
        query: str,
    ) -> list[ArgumentNode]:
        """クエリに最も関連する上位 top_k ノードを返す。

        フォールバック動作:
        - ノード数 ≤ top_k → 全て返す（スコア計算不要）
        - クエリのトークンが空 → 先頭 top_k を返す
        - 全スコアが 0   → 先頭 top_k を返す（クエリ語がコーパスに存在しない場合）

        Args:
            nodes: 候補となるノードリスト（同一クエリ型の入口候補）。
            query: ユーザーのクエリ文字列。

        Returns:
            BM25 スコア降順の上位 top_k ノード。
        """
        if len(nodes) <= self._top_k:
            return nodes

        query_tokens = _tokenize(query)
        if not query_tokens:
            return nodes[: self._top_k]

        scored = self._score_all(nodes, query_tokens)

        # 全スコアが 0 → クエリ語がノードテキストに一切存在しない
        # フォールバック: 元のストア順で先頭 top_k を返す
        if all(s.score == 0.0 for s in scored):
            return nodes[: self._top_k]

        scored.sort(key=lambda s: s.score, reverse=True)
        return [s.node for s in scored[: self._top_k]]

    # ── BM25 スコア計算 ────────────────────────────────────────────────────────

    def _score_all(
        self,
        nodes: list[ArgumentNode],
        query_tokens: list[str],
    ) -> list[_ScoredNode]:
        """全ノードに BM25 スコアを付けて返す。

        BM25 公式:
            score(d, Q) = Σ_q IDF(q) * tf(q,d)*(k1+1) / (tf(q,d) + k1*(1-b+b*|d|/avgdl))
            IDF(q)      = log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)
        """
        # ── 前処理 ──────────────────────────────────────────────────────────────
        # パラフレーズで固有名詞・数値が消えないよう、verbatim_text を優先。
        tokenized: list[list[str]] = [
            _tokenize(getattr(n, "verbatim_text", None) or n.text) for n in nodes
        ]
        term_freqs: list[Counter[str]] = [Counter(toks) for toks in tokenized]
        doc_lens: list[int] = [len(toks) for toks in tokenized]
        avgdl: float = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
        n_docs: int = len(nodes)

        # ── IDF（逆文書頻度）──────────────────────────────────────────────────
        idf: dict[str, float] = {}
        for qt in set(query_tokens):
            df = sum(1 for tf in term_freqs if tf[qt] > 0)
            # Robertson-Walker IDF（スムージングあり、負値なし）
            idf[qt] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        # ── 各ノードのスコア計算 ────────────────────────────────────────────────
        results: list[_ScoredNode] = []
        for idx, node in enumerate(nodes):
            score = 0.0
            dl = doc_lens[idx]
            tf_map = term_freqs[idx]
            norm = 1.0 - self._b + self._b * dl / avgdl  # 文書長正規化係数

            for qt in query_tokens:
                tf = tf_map[qt]
                if tf == 0:
                    continue
                tf_norm = tf * (self._k1 + 1.0) / (tf + self._k1 * norm)
                score += idf[qt] * tf_norm

            results.append(_ScoredNode(node=node, score=score))

        return results
