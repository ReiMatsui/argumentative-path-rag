"""
文書をセンテンスレベルのチャンクに分割するモジュール。

設計方針:
- チャンクはオーバーラップなしのセンテンス境界で区切る（議論単位を壊さないため）
- 最大トークン数を超えたチャンクは自動的に分割する
- 将来的に tiktoken 以外のトークナイザに差し替えられるよう抽象化する
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    """1チャンクを表す不変値オブジェクト。

    Attributes:
        doc_id: 元文書の識別子。
        chunk_idx: 文書内での0-basedインデックス。
        text: チャンクのテキスト内容。
        char_start: 元文書内の開始文字位置。
        char_end: 元文書内の終了文字位置。
        metadata: 任意の追加情報。
    """

    doc_id: str
    chunk_idx: int
    text: str
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


class SentenceChunker:
    """センテンス境界でテキストをチャンク分割する。

    Args:
        max_tokens: 1チャンクあたりの最大トークン数。
        sentence_sep_pattern: センテンス境界を検出する正規表現。
    """

    # 日本語・英語混在を考慮したセンテンス区切りパターン
    _DEFAULT_PATTERN = r"(?<=[。！？.!?])\s*"

    def __init__(
        self,
        max_tokens: int = 256,
        sentence_sep_pattern: str = _DEFAULT_PATTERN,
    ) -> None:
        self.max_tokens = max_tokens
        self._sep_pattern = re.compile(sentence_sep_pattern)

    def chunk(self, text: str, doc_id: str) -> list[DocumentChunk]:
        """テキストをチャンクに分割して返す。

        Args:
            text: 分割対象の生テキスト。
            doc_id: 元文書の識別子。

        Returns:
            DocumentChunk のリスト（chunk_idx は 0 始まり）。
        """
        sentences = self._split_sentences(text)
        return self._merge_into_chunks(sentences, doc_id)

    # ── private ────────────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """テキストをセンテンスに分割し、(text, start, end) のタプルで返す。"""
        spans: list[tuple[str, int, int]] = []
        pos = 0
        for part in self._sep_pattern.split(text):
            part = part.strip()
            if not part:
                pos += len(part)
                continue
            start = text.find(part, pos)
            end = start + len(part)
            spans.append((part, start, end))
            pos = end
        return spans

    def _count_tokens(self, text: str) -> int:
        """簡易トークン数推定（文字数 / 2 で近似）。

        本番環境では tiktoken に差し替える。
        """
        return len(text) // 2

    def _merge_into_chunks(
        self,
        sentences: list[tuple[str, int, int]],
        doc_id: str,
    ) -> list[DocumentChunk]:
        """センテンスをトークン上限を守りながらチャンクに結合する。"""
        chunks: list[DocumentChunk] = []
        current_sentences: list[tuple[str, int, int]] = []
        current_tokens = 0

        def flush(idx: int) -> None:
            if not current_sentences:
                return
            merged = " ".join(s for s, _, _ in current_sentences)
            char_start = current_sentences[0][1]
            char_end = current_sentences[-1][2]
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    chunk_idx=idx,
                    text=merged,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            current_sentences.clear()

        chunk_idx = 0
        for sent, start, end in sentences:
            tokens = self._count_tokens(sent)
            if current_tokens + tokens > self.max_tokens and current_sentences:
                flush(chunk_idx)
                chunk_idx += 1
                current_tokens = 0
            current_sentences.append((sent, start, end))
            current_tokens += tokens

        flush(chunk_idx)
        return chunks
