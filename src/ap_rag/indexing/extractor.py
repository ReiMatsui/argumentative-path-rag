"""
LLM を使ったエッジ抽出モジュール。

OpenAI Structured Outputs を使い、EdgeType 列挙値以外を API レベルで排除する。
JSON パースは不要。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.indexing.schemas import EdgeExtractionOutput
from ap_rag.models.graph import ArgumentEdge, ArgumentNode
from ap_rag.openai_compat import reasoning_kwarg, sampling_kwargs

logger = logging.getLogger(__name__)

# ── プロンプト定義 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in argument analysis. Given a list of argumentative nodes extracted \
from the same document, identify the directed relationships between them.

Edge type definitions and direction rules:

- SUPPORTS   : (EVIDENCE/CONCLUSION) ──SUPPORTS──▶ (CLAIM)
               The source provides empirical or logical support for the target claim.
               Example: "Measured outcomes improved by a clear margin across trials"
                        SUPPORTS "The proposed method is effective."

- CONTRADICTS: (CONTRAST) ──CONTRADICTS──▶ (CLAIM)
               The source challenges or refutes the target.
               Example: "Removing component X noticeably degrades performance"
                        CONTRADICTS "Component X is not essential."

- DERIVES    : (EVIDENCE) ──DERIVES──▶ (CONCLUSION)
               The target conclusion is derived or inferred from the source evidence.
               Example: "Outcomes drop sharply when condition A is absent"
                        DERIVES "Condition A is critical for the outcome."

- ASSUMES    : (CLAIM or CONCLUSION) ──ASSUMES──▶ (ASSUMPTION)
               The source claim/conclusion only holds if the target assumption is true.
               CRITICAL: source must be CLAIM or CONCLUSION; target must be ASSUMPTION.
               Example: "The result generalizes to new settings" ASSUMES
                        "The sample is representative of those settings."

- ILLUSTRATES: (EVIDENCE) ──ILLUSTRATES──▶ (CLAIM)
               The source is a concrete example or sample that illustrates the target.
               Example: "The case study in Section 3 shows behavior B in practice"
                        ILLUSTRATES "The system exhibits behavior B."

- CONTRASTS  : (CONTRAST) ──CONTRASTS──▶ (CLAIM)
               The source presents a contrasting case relative to the target.

Rules:
1. Each node is provided with an ``idx``, a ``type``, and two text fields:
   ``text`` (short label) and ``source_span`` (verbatim original sentence). When
   deciding relations, ground your reasoning in ``source_span`` to avoid being
   misled by paraphrased labels.
2. Only output edges with confidence >= 0.7.
3. Strictly follow the direction conventions above — especially for ASSUMES.
4. Do not create edges between identical texts or self-loops (source_idx != target_idx).
5. Prefer recall over precision for within-same-topic pairs: if two nodes clearly
   discuss the same subject, surface a SUPPORTS or DERIVES edge even if the
   wording is indirect.
6. If there are no clear relationships, return an empty edges list.
"""

_USER_TEMPLATE = """\
Nodes:
{nodes_json}

Identify all directed argumentative relationships between these nodes.
"""


class EdgeExtractor:
    """ノードのリストを受け取り、エッジ候補を返す。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名（.env の OPENAI_CLASSIFIER_MODEL で切り替え可）。
        min_confidence: この値未満のエッジは除外する。
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o",
        min_confidence: float = 0.7,
    ) -> None:
        self._client = client
        self._model = model
        self._min_confidence = min_confidence

    def extract(self, nodes: list[ArgumentNode]) -> list[ArgumentEdge]:
        """ノードリストからエッジを抽出する。

        ノードが2つ未満の場合は空リストを返す。
        """
        if len(nodes) < 2:
            return []

        nodes_json = json.dumps(
            [
                {
                    "idx": i,
                    "type": n.node_type.value,
                    "text": n.text,
                    "source_span": (n.source_span or n.text),
                }
                for i, n in enumerate(nodes)
            ],
            ensure_ascii=False,
            indent=2,
        )
        output = self._call_llm(nodes_json)
        return self._to_edges(output, nodes)

    # ── LLM 呼び出し ──────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, nodes_json: str) -> EdgeExtractionOutput:
        """Structured Outputs で EdgeExtractionOutput を返す。"""
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(nodes_json=nodes_json)},
            ],
            response_format=EdgeExtractionOutput,
            **sampling_kwargs(self._model, temperature=0.0),
            **reasoning_kwarg(self._model),
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("Structured Outputs: parsed が None（refusal の可能性）")
            return EdgeExtractionOutput(edges=[])
        return parsed

    # ── 変換 ──────────────────────────────────────────────────────────────────

    def _to_edges(
        self,
        output: EdgeExtractionOutput,
        nodes: list[ArgumentNode],
    ) -> list[ArgumentEdge]:
        """EdgeExtractionOutput → ArgumentEdge のリストに変換する。"""
        edges: list[ArgumentEdge] = []
        for item in output.edges:
            src_idx, tgt_idx = item.source_idx, item.target_idx

            if src_idx == tgt_idx:
                continue
            if not (0 <= src_idx < len(nodes) and 0 <= tgt_idx < len(nodes)):
                logger.warning("範囲外インデックス: src=%d tgt=%d (nodes=%d)", src_idx, tgt_idx, len(nodes))
                continue
            if item.confidence < self._min_confidence:
                continue

            edges.append(
                ArgumentEdge(
                    edge_type=item.edge_type,
                    source_id=nodes[src_idx].id,
                    target_id=nodes[tgt_idx].id,
                    confidence=item.confidence,
                )
            )

        logger.debug("エッジ抽出: %d 件", len(edges))
        return edges
