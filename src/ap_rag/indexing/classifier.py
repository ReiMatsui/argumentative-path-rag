"""
LLM を使ったノード分類モジュール。

1チャンクから 1つ以上の ArgumentNode を生成する。
LLM は JSON 形式で {node_type, text} の配列を返すよう指示する。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.indexing.chunker import DocumentChunk
from ap_rag.models.graph import ArgumentNode
from ap_rag.models.taxonomy import NodeType

logger = logging.getLogger(__name__)

# ── プロンプト定義 ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in argument mining. Given a text passage, identify all argumentative \
units and classify each into exactly one of the following node types:

- CLAIM: A central assertion or proposition.
- EVIDENCE: Empirical data, statistics, or citations that support a claim.
- ASSUMPTION: An implicit premise that a claim depends on.
- CONCLUSION: A conclusion drawn from reasoning or evidence.
- CAVEAT: A qualification, exception, or limitation on a claim.
- CONTRAST: A counter-example or contrasting statement.
- DEFINITION: A definition of a term or concept.

Rules:
1. Extract one node per distinct argumentative unit (sentence or clause).
2. Output ONLY a JSON object with a single key "nodes" containing an array of objects.
3. Each object must have exactly two keys: "node_type" and "text".
4. If a sentence contains multiple argumentative units, split them into separate entries.

Required output format:
{"nodes": [
  {"node_type": "CLAIM", "text": "Q3 revenue declined 12% year-over-year."},
  {"node_type": "EVIDENCE", "text": "Inventory adjustments in electronic components were the primary factor."}
]}
"""

_USER_TEMPLATE = "Passage:\n{text}"


class NodeClassifier:
    """チャンクを受け取り、ノード候補のリストを返す。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名。
        max_retries: LLM 失敗時の最大リトライ回数。
    """

    def __init__(
        self,
        client: Any,  # openai.OpenAI — 型ヒントは実行時にインポート
        model: str = "gpt-4o",
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self._model = model
        self._max_retries = max_retries

    def classify(self, chunk: DocumentChunk) -> list[ArgumentNode]:
        """1チャンクから ArgumentNode のリストを生成する。

        Args:
            chunk: 分類対象のチャンク。

        Returns:
            抽出されたノードのリスト（空の場合もある）。
        """
        raw = self._call_llm(chunk.text)
        return self._parse_response(raw, chunk)

    # ── LLM呼び出し ───────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, text: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(text=text)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return content

    # ── レスポンス解析 ────────────────────────────────────────────────────────

    def _parse_response(
        self,
        raw: str,
        chunk: DocumentChunk,
    ) -> list[ArgumentNode]:
        """LLM の JSON レスポンスを ArgumentNode のリストに変換する。"""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "JSONデコード失敗。chunk_idx=%d, raw=%s...",
                chunk.chunk_idx,
                raw[:200],
            )
            return []

        # dictの場合: "nodes"キーを優先し、なければ最初のlist値を使う
        if isinstance(data, dict):
            if "nodes" in data:
                data = data["nodes"]
            else:
                # どのキーにリストが入っているか探す
                list_values = [v for v in data.values() if isinstance(v, list)]
                if list_values:
                    logger.debug("'nodes'キーなし。最初のlist値を使用: keys=%s", list(data.keys()))
                    data = list_values[0]
                else:
                    logger.warning("dictにlistが見つからない: keys=%s, raw=%s...", list(data.keys()), raw[:300])
                    return []

        if not isinstance(data, list):
            logger.warning("予期しない形式のレスポンス: %s, raw=%s...", type(data), raw[:300])
            return []

        nodes: list[ArgumentNode] = []
        for item in data:
            node = self._item_to_node(item, chunk)
            if node is not None:
                nodes.append(node)

        logger.debug(
            "chunk_idx=%d → %d ノード抽出", chunk.chunk_idx, len(nodes)
        )
        return nodes

    def _item_to_node(
        self,
        item: dict[str, Any],
        chunk: DocumentChunk,
    ) -> ArgumentNode | None:
        """1アイテムを ArgumentNode に変換する。失敗時は None を返す。"""
        try:
            raw_type = str(item.get("node_type", "")).upper()
            node_type = NodeType(raw_type)
            text = str(item.get("text", "")).strip()
            if not text:
                return None
            return ArgumentNode(
                node_type=node_type,
                text=text,
                source_doc_id=chunk.doc_id,
                source_chunk_idx=chunk.chunk_idx,
            )
        except (ValueError, KeyError) as e:
            logger.warning("ノード変換失敗: %s — item=%s", e, item)
            return None
