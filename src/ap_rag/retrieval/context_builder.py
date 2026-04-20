"""
役割付きコンテキスト構築モジュール。

グラフ探索で収集したノードを、LLMへ渡すための
構造化されたコンテキスト文字列に変換する。

ノードタイプごとにセクションを分けて提示することで
LLMが役割を意識した回答を生成できるようにする。
"""

from __future__ import annotations

from dataclasses import dataclass

from ap_rag.models.graph import ArgumentNode
from ap_rag.models.taxonomy import NodeType, QueryType

# ── 表示ラベル（日英）────────────────────────────────────────────────────────

_NODE_TYPE_LABELS: dict[NodeType, str] = {
    NodeType.CLAIM:      "【主張】",
    NodeType.EVIDENCE:   "【根拠】",
    NodeType.ASSUMPTION: "【前提】",
    NodeType.CONCLUSION: "【結論】",
    NodeType.CAVEAT:     "【留保】",
    NodeType.CONTRAST:   "【対比】",
    NodeType.DEFINITION: "【定義】",
}

# クエリ型ごとの優先表示順
_NODE_TYPE_ORDER: dict[QueryType, list[NodeType]] = {
    QueryType.WHY: [
        NodeType.EVIDENCE, NodeType.CONCLUSION, NodeType.ASSUMPTION,
        NodeType.CLAIM, NodeType.CAVEAT,
    ],
    QueryType.WHAT: [
        NodeType.CLAIM, NodeType.DEFINITION, NodeType.EVIDENCE,
    ],
    QueryType.HOW: [
        NodeType.CLAIM, NodeType.EVIDENCE, NodeType.CONCLUSION,
    ],
    QueryType.EVIDENCE: [
        NodeType.EVIDENCE, NodeType.CLAIM,
    ],
    QueryType.ASSUMPTION: [
        NodeType.ASSUMPTION, NodeType.CLAIM, NodeType.CONCLUSION,
    ],
}


@dataclass
class RetrievalContext:
    """検索結果を格納するデータクラス。

    Attributes:
        query: 元のクエリ文字列。
        query_type: 判定されたクエリ型。
        nodes: 収集されたノード（型でソート済み）。
        context_text: LLMに渡す整形済みコンテキスト文字列。
    """

    query: str
    query_type: QueryType
    nodes: list[ArgumentNode]
    context_text: str

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


class ContextBuilder:
    """ノードリストから役割付きコンテキスト文字列を生成する。

    Args:
        max_nodes: コンテキストに含める最大ノード数。
    """

    def __init__(self, max_nodes: int = 20) -> None:
        self._max_nodes = max_nodes

    def build(
        self,
        query: str,
        query_type: QueryType,
        nodes: list[ArgumentNode],
    ) -> RetrievalContext:
        """ノードを整理してコンテキストを構築する。

        Args:
            query: ユーザーのクエリ。
            query_type: クエリ型。
            nodes: グラフ探索で収集したノード。

        Returns:
            RetrievalContext。
        """
        sorted_nodes = self._sort_nodes(nodes, query_type)
        trimmed = sorted_nodes[: self._max_nodes]
        context_text = self._format(query, query_type, trimmed)
        return RetrievalContext(
            query=query,
            query_type=query_type,
            nodes=trimmed,
            context_text=context_text,
        )

    # ── private ────────────────────────────────────────────────────────────

    def _sort_nodes(
        self,
        nodes: list[ArgumentNode],
        query_type: QueryType,
    ) -> list[ArgumentNode]:
        """クエリ型に応じた優先順でノードをソートする。"""
        order = _NODE_TYPE_ORDER.get(query_type, list(NodeType))
        type_priority = {nt: i for i, nt in enumerate(order)}

        return sorted(
            nodes,
            key=lambda n: type_priority.get(n.node_type, len(order)),
        )

    def _format(
        self,
        query: str,
        query_type: QueryType,
        nodes: list[ArgumentNode],
    ) -> str:
        """LLMに渡す整形済みテキストを生成する。"""
        if not nodes:
            return "（関連する情報が見つかりませんでした）"

        lines: list[str] = [
            f"クエリ: {query}",
            f"クエリタイプ: {query_type.value}",
            "",
            "--- 参照情報 ---",
        ]

        # ノードタイプごとにグルーピングして表示
        grouped: dict[NodeType, list[ArgumentNode]] = {}
        for node in nodes:
            grouped.setdefault(node.node_type, []).append(node)

        order = _NODE_TYPE_ORDER.get(query_type, list(NodeType))
        for node_type in order:
            if node_type not in grouped:
                continue
            label = _NODE_TYPE_LABELS[node_type]
            for node in grouped[node_type]:
                lines.append(f"{label} {node.text}")

        lines.append("--- 参照情報ここまで ---")
        return "\n".join(lines)
