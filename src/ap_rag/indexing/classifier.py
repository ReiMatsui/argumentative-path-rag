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

logger = logging.getLogger(__name__)

# ── プロンプト定義 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in argument mining. Given a text passage, identify all argumentative \
units and classify each into exactly one of the following node types:

- CLAIM: A central assertion, proposition, or research question being investigated.
- EVIDENCE: Empirical data, statistics, experimental results, citations, or examples \
that support or illustrate a claim.
- ASSUMPTION: An implicit or explicit premise that a claim or conclusion depends on.
- CONCLUSION: A conclusion drawn from reasoning, evidence, or experiments.
- CAVEAT: A qualification, exception, limitation, or boundary condition on a claim.
- CONTRAST: A counter-example, competing result, or contrasting statement.
- DEFINITION: A definition of a term, concept, or notation.

Rules:
1. Extract one node per distinct argumentative unit (sentence or clause).
2. If a sentence contains multiple argumentative units, split them into separate entries.
3. Map ambiguous units to the closest type:
   - Questions / hypotheses → CLAIM
   - Examples / results / findings → EVIDENCE
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
            temperature=0.0,
            response_format=NodeClassificationOutput,
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
        """NodeClassificationOutput → ArgumentNode のリストに変換する。"""
        nodes: list[ArgumentNode] = []
        for item in output.nodes:
            text = item.text.strip()
            if not text:
                continue
            nodes.append(
                ArgumentNode(
                    node_type=item.node_type,
                    text=text,
                    source_doc_id=chunk.doc_id,
                    source_chunk_idx=chunk.chunk_idx,
                )
            )
        logger.debug("chunk_idx=%d → %d ノード抽出", chunk.chunk_idx, len(nodes))
        return nodes
