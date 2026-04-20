"""
LLM を使ったエッジ抽出モジュール。

ノードのペアに対して、関係の有無と種別を判定する。
コスト削減のため、チャンク内のノードペアのみを対象とする（局所エッジ）。
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.models.graph import ArgumentEdge, ArgumentNode
from ap_rag.models.taxonomy import EdgeType

logger = logging.getLogger(__name__)

# ── プロンプト定義 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in argument analysis. Given a list of argumentative nodes, \
identify the directed relationships between them.

Edge type definitions and direction rules:

- SUPPORTS   : (EVIDENCE/CONCLUSION) ──SUPPORTS──▶ (CLAIM)
               The source provides empirical or logical support for the target claim.
               Example: "Inventory adjustments were the main factor" SUPPORTS "Q3 revenue fell 12%"

- CONTRADICTS: (CONTRAST) ──CONTRADICTS──▶ (CLAIM)
               The source challenges or refutes the target.
               Example: "Competitor grew 8%" CONTRADICTS "Our sales declined"

- DERIVES    : (EVIDENCE) ──DERIVES──▶ (CONCLUSION)
               The target conclusion is derived/inferred from the source evidence.
               Example: "Semiconductor shortage caused stoppage" DERIVES "Shortage will continue next quarter"

- ASSUMES    : (CLAIM or CONCLUSION) ──ASSUMES──▶ (ASSUMPTION)
               The source claim/conclusion only holds if the target assumption is true.
               The source DEPENDS ON the target assumption as a hidden premise.
               Example: "Q3 revenue fell 12%" ASSUMES "Exchange rates were stable (so FX is not the cause)"
               CRITICAL: source must be CLAIM or CONCLUSION; target must be ASSUMPTION.

- ILLUSTRATES: (visual/example node) ──ILLUSTRATES──▶ (CLAIM)
               The source concretely illustrates or exemplifies the target.

- CONTRASTS  : (CONTRAST) ──CONTRASTS──▶ (CLAIM)
               The source presents a contrasting case relative to the target.

Rules:
1. Only output edges with confidence >= 0.7.
2. Strictly follow the direction conventions above — especially for ASSUMES:
   source = CLAIM or CONCLUSION, target = ASSUMPTION.
3. Output ONLY a JSON object with a single key "edges" containing an array of objects.
4. Each object must have keys: "source_idx", "target_idx", "edge_type", "confidence"
   where idx refers to the 0-based index in the provided nodes list.
5. If there are no clear relationships, output: {"edges": []}
6. Do not create edges between identical texts.

Required output format:
{"edges": [
  {"source_idx": 0, "target_idx": 2, "edge_type": "ASSUMES", "confidence": 0.85},
  {"source_idx": 1, "target_idx": 0, "edge_type": "SUPPORTS", "confidence": 0.95}
]}
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
        model: 使用するモデル名。
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

        Args:
            nodes: 同一チャンク内のノードリスト（順序は chunk_idx でソート済みを想定）。

        Returns:
            抽出された ArgumentEdge のリスト。
        """
        if len(nodes) < 2:
            return []

        nodes_json = json.dumps(
            [{"idx": i, "type": n.node_type.value, "text": n.text} for i, n in enumerate(nodes)],
            ensure_ascii=False,
            indent=2,
        )
        raw = self._call_llm(nodes_json)
        return self._parse_response(raw, nodes)

    # ── LLM呼び出し ───────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, nodes_json: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(nodes_json=nodes_json)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    # ── レスポンス解析 ────────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        nodes: list[ArgumentNode],
    ) -> list[ArgumentEdge]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("EdgeExtractor: JSONデコード失敗 raw=%s...", raw[:200])
            return []

        # dictの場合: "edges"キーを優先し、なければ最初のlist値を使う
        if isinstance(data, dict):
            if "edges" in data:
                data = data["edges"]
            else:
                list_values = [v for v in data.values() if isinstance(v, list)]
                if list_values:
                    data = list_values[0]
                else:
                    return []

        if not isinstance(data, list):
            return []

        edges: list[ArgumentEdge] = []
        for item in data:
            edge = self._item_to_edge(item, nodes)
            if edge is not None:
                edges.append(edge)

        logger.debug("エッジ抽出: %d 件", len(edges))
        return edges

    def _item_to_edge(
        self,
        item: dict[str, Any],
        nodes: list[ArgumentNode],
    ) -> ArgumentEdge | None:
        try:
            src_idx = int(item["source_idx"])
            tgt_idx = int(item["target_idx"])
            confidence = float(item.get("confidence", 1.0))
            edge_type = EdgeType(str(item["edge_type"]).upper())

            if src_idx == tgt_idx:
                return None
            if not (0 <= src_idx < len(nodes) and 0 <= tgt_idx < len(nodes)):
                logger.warning("範囲外インデックス: src=%d tgt=%d", src_idx, tgt_idx)
                return None
            if confidence < self._min_confidence:
                return None

            return ArgumentEdge(
                edge_type=edge_type,
                source_id=nodes[src_idx].id,
                target_id=nodes[tgt_idx].id,
                confidence=confidence,
            )
        except (KeyError, ValueError, IndexError) as e:
            logger.warning("エッジ変換失敗: %s — item=%s", e, item)
            return None
