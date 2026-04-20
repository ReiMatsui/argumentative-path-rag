"""
LLM を使ったノード分類モジュール。

OpenAI Structured Outputs を使い、NodeType 列挙値以外を API レベルで排除する。
JSON パースやフォールバックマッピングは不要。
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.indexing.chunker import DocumentChunk
from ap_rag.indexing.schemas import NodeClassificationOutput
from ap_rag.models.graph import ArgumentNode
from ap_rag.openai_compat import reasoning_kwarg, sampling_kwargs

logger = logging.getLogger(__name__)

# ── プロンプト定義 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in argument mining. Given a text passage, extract a complete, \
lossless set of nodes covering BOTH argumentative units AND concrete factual \
statements, then classify each by its argumentative role.

Node types:
- CLAIM: A central assertion, proposition, question, or hypothesis being argued for.
- EVIDENCE: Any factual, empirical, or concrete supporting material — data, results, \
examples, citations, specific numbers, named entities, descriptive facts, procedural \
details. Even if a sentence looks purely descriptive rather than overtly argumentative, \
keep it as EVIDENCE so downstream factual questions can still retrieve it.
- ASSUMPTION: An implicit or explicit premise that a claim or conclusion depends on.
- CONCLUSION: A conclusion drawn from reasoning, evidence, or prior statements.
- CAVEAT: A qualification, exception, limitation, or boundary condition on a claim.
- CONTRAST: A counter-example, competing position, or contrasting statement.
- DEFINITION: A definition of a term, concept, or named entity.

Coverage rule (VERY IMPORTANT):
  Prefer over-extraction to under-extraction. If you are unsure whether a sentence \
  is argumentative, STILL include it as an EVIDENCE node. The downstream system can \
  filter, but it cannot recover information you dropped. **Every sentence that carries \
  factual, definitional, or argumentative content should produce at least one node.**

Splitting rule:
  If a sentence contains multiple distinct units (e.g., a CLAIM + its EVIDENCE in one \
  sentence), split them into separate entries.

Output rule (VERY IMPORTANT):
  For each node, fill TWO fields:
    - source_span: the VERBATIM substring of the input passage covering this unit. \
      Copy it exactly — same case, same punctuation, same numbers, same named entities. \
      DO NOT paraphrase, translate, or normalize in this field. This is the lossless \
      original that downstream retrieval and evaluation depend on.
    - text: a short normalized label (<= 200 chars) used only for display. You may \
      lightly paraphrase here for readability.
  If source_span and the cleanest paraphrase are identical, copy the same string \
  into both fields.

Ambiguity fallback:
  - Questions / hypotheses → CLAIM
  - Examples / results / findings / concrete facts → EVIDENCE
  - Implications → CONCLUSION
  - Limitations / restrictions → CAVEAT
"""

_USER_TEMPLATE = "Passage:\n{text}"


class NodeClassifier:
    """チャンクを受け取り、ノード候補のリストを返す。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名（.env の OPENAI_CLASSIFIER_MODEL で切り替え可）。
        max_retries: LLM 失敗時の最大リトライ回数。
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o",
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries

    def classify(self, chunk: DocumentChunk) -> list[ArgumentNode]:
        """1チャンクから ArgumentNode のリストを生成する。"""
        output = self._call_llm(chunk.text)
        return self._to_nodes(output, chunk)

    # ── LLM 呼び出し ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, text: str) -> NodeClassificationOutput:
        """Structured Outputs で NodeClassificationOutput を返す。"""
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
            ],
            response_format=NodeClassificationOutput,
            **sampling_kwargs(self._model, temperature=0.0),
            **reasoning_kwarg(self._model),
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("Structured Outputs: parsed が None（refusal の可能性）")
            return NodeClassificationOutput(nodes=[])
        return parsed

    # ── 変換 ──────────────────────────────────────────────────────────────────

    def _to_nodes(
        self,
        output: NodeClassificationOutput,
        chunk: DocumentChunk,
    ) -> list[ArgumentNode]:
        """NodeClassificationOutput → ArgumentNode のリストに変換する。

        LLM が返してきた ``source_span`` を元チャンクに対して検証する:
          - チャンク本文に部分一致すればそのまま verbatim として採用
          - 一致しなければパラフレーズされた可能性が高いのでログに警告を残し、
            text と source_span の "長い方" を保持する（情報欠落を避ける）
        """
        nodes: list[ArgumentNode] = []
        chunk_text = chunk.text
        for item in output.nodes:
            text = item.text.strip()
            span = (item.source_span or "").strip()
            if not text and not span:
                continue
            # 検証: span が passage に実在するか
            if span and span not in chunk_text:
                # 完全一致しなければ警告。ただし情報量のある方を優先的に残す。
                logger.warning(
                    "chunk_idx=%d: source_span が passage に実在しません (len=%d). "
                    "LLM がパラフレーズした可能性。span の先頭: %r",
                    chunk.chunk_idx, len(span), span[:80],
                )
                # 片方でも生存させる: span の方が長ければそれを採用、短ければ text を採用
                if len(span) < len(text):
                    span = ""
            # text が空なら span をラベルに流用（必ず何かを残す）
            if not text and span:
                text = span[:200]
            nodes.append(
                ArgumentNode(
                    node_type=item.node_type,
                    text=text,
                    source_span=span if span else None,
                    source_doc_id=chunk.doc_id,
                    source_chunk_idx=chunk.chunk_idx,
                )
            )
        logger.debug("chunk_idx=%d → %d ノード抽出", chunk.chunk_idx, len(nodes))
        return nodes
