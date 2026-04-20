"""
比較ベースラインのRAGシステム実装。

研究計画書 §4.3「比較ベースライン」:
  - BM25RAG     : 標準RAG（弱）— キーワードマッチ
  - DenseRAG    : 標準RAG（強）— 密ベクトル検索（E5-Mistral-7B）

両クラスは ArgumentativeRAGPipeline と同じ .query() インターフェースを持つ。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ap_rag.openai_compat import reasoning_kwarg, sampling_kwargs

logger = logging.getLogger(__name__)


# ── ベースライン共通の戻り値型 ────────────────────────────────────────────────

@dataclass
class BaselineResult:
    """ベースラインRAGの回答結果。

    ArgumentativeRAGPipeline の GenerationResult と互換の shape にする。
    """

    answer: str
    retrieval_context: _BaselineContext

    @dataclass
    class _BaselineContext:
        nodes: list[Any]  # テキストを持つ疑似ノード


@dataclass
class _TextNode:
    """ベースライン用の簡易テキストノード。"""
    text: str


# ── BM25ベースライン ──────────────────────────────────────────────────────────

class BM25RAG:
    """BM25 を使ったシンプルなキーワードベースRAG。

    Args:
        client: openai.OpenAI インスタンス（回答生成用）。
        model: 回答生成に使うモデル名。
        top_k: 検索で返すチャンク数。
        k1: BM25 パラメータ（term frequency の飽和度）。
        b: BM25 パラメータ（文書長正規化係数）。
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._client = client
        self._model = model
        self._top_k = top_k
        self._k1 = k1
        self._b = b
        # doc_id → list[str]（チャンクテキスト）
        self._index: dict[str, list[str]] = {}

    def index(self, texts: list[str], doc_id: str) -> None:
        """文書テキストをインデックスする。"""
        self._index[doc_id] = texts

    def query(self, question: str, doc_id: str) -> BaselineResult:
        """BM25 で検索し、LLMで回答を生成する。"""
        chunks = self._index.get(doc_id, [])
        if not chunks:
            return self._empty_result()

        scores = self._bm25_scores(question, chunks)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_chunks = [chunks[i] for i in top_indices[: self._top_k]]

        answer = self._generate(question, top_chunks)
        nodes = [_TextNode(text=c) for c in top_chunks]
        return BaselineResult(
            answer=answer,
            retrieval_context=BaselineResult._BaselineContext(nodes=nodes),
        )

    def _bm25_scores(self, query: str, chunks: list[str]) -> list[float]:
        """BM25 スコアを計算して返す。"""
        query_tokens = self._tokenize(query)
        tokenized_chunks = [self._tokenize(c) for c in chunks]
        avg_dl = sum(len(t) for t in tokenized_chunks) / max(len(tokenized_chunks), 1)

        scores: list[float] = []
        n = len(chunks)
        for tokens in tokenized_chunks:
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in query_tokens:
                df = sum(1 for t in tokenized_chunks if term in t)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_val = tf.get(term, 0)
                numerator = tf_val * (self._k1 + 1)
                denominator = tf_val + self._k1 * (1 - self._b + self._b * dl / avg_dl)
                score += idf * numerator / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """簡易トークナイズ（英数字・日本語文字で分割）。"""
        text = text.lower()
        return re.findall(r"[a-z0-9]+|[^\x00-\x7f]", text)

    def _generate(self, question: str, contexts: list[str]) -> str:
        context_text = "\n".join(f"- {c}" for c in contexts)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided context. Be concise.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            **sampling_kwargs(self._model, temperature=0.1),
            **reasoning_kwarg(self._model),
        )
        return (response.choices[0].message.content or "").strip()

    def _empty_result(self) -> BaselineResult:
        return BaselineResult(
            answer="",
            retrieval_context=BaselineResult._BaselineContext(nodes=[]),
        )


# ── Dense（埋め込み）ベースライン ─────────────────────────────────────────────

class DenseRAG:
    """密ベクトル検索を使った標準RAG（強）。

    SentenceTransformer でチャンクを埋め込み、コサイン類似度で検索する。
    本番評価では intfloat/e5-mistral-7b-instruct を使用する。
    軽量モデル（all-MiniLM-L6-v2 など）でも動作する。

    Args:
        client: openai.OpenAI インスタンス（回答生成用）。
        model: 回答生成に使うモデル名。
        embedding_model: SentenceTransformer のモデル名。
        top_k: 検索で返すチャンク数。
        device: 実行デバイス（"cpu" / "cuda" / "mps"）。
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        device: str = "cpu",
        encoder: Any | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._top_k = top_k
        self._device = device
        self._embedding_model_name = embedding_model
        # encoder が注入された場合は torch/sentence-transformers 依存を回避できる
        self._encoder: Any = encoder
        # doc_id → (chunks, embeddings) — embeddings は numpy.ndarray
        self._index: dict[str, tuple[list[str], Any]] = {}

    def _get_encoder(self) -> Any:
        """SentenceTransformer を遅延ロードする。"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(
                self._embedding_model_name, device=self._device
            )
        return self._encoder

    def index(self, texts: list[str], doc_id: str) -> None:
        """文書テキストをエンコードしてインデックスする。

        埋め込みは正規化済み numpy.ndarray として保持し、
        検索時は内積 = コサイン類似度で順位付けする。
        """
        import numpy as np

        encoder = self._get_encoder()
        # 正規化まで encoder 側で済ませる（torch 依存を避ける）
        embeddings = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._index[doc_id] = (texts, embeddings)

    def query(self, question: str, doc_id: str) -> BaselineResult:
        """コサイン類似度で検索し、LLMで回答を生成する。"""
        import numpy as np

        if doc_id not in self._index:
            return self._empty_result()

        chunks, embeddings = self._index[doc_id]
        encoder = self._get_encoder()

        query_emb = encoder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        # embeddings は (N, D) で正規化済み → 内積がコサイン類似度
        scores = embeddings @ query_emb
        top_indices = np.argsort(-scores)[: self._top_k].tolist()
        top_chunks = [chunks[i] for i in top_indices]

        answer = self._generate(question, top_chunks)
        nodes = [_TextNode(text=c) for c in top_chunks]
        return BaselineResult(
            answer=answer,
            retrieval_context=BaselineResult._BaselineContext(nodes=nodes),
        )

    def _generate(self, question: str, contexts: list[str]) -> str:
        context_text = "\n".join(f"- {c}" for c in contexts)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the provided context. Be concise.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            **sampling_kwargs(self._model, temperature=0.1),
            **reasoning_kwarg(self._model),
        )
        return (response.choices[0].message.content or "").strip()

    def _empty_result(self) -> BaselineResult:
        return BaselineResult(
            answer="",
            retrieval_context=BaselineResult._BaselineContext(nodes=[]),
        )
