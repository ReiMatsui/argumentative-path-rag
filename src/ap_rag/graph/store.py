"""
グラフストアの抽象インターフェース。

Neo4j と NetworkX の両実装に共通のAPIを提供する。
将来的に他のグラフDBへの差し替えもこのインターフェースを実装するだけでよい。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ap_rag.models.graph import ArgumentEdge, ArgumentGraph, ArgumentNode
from ap_rag.models.taxonomy import NodeType


class GraphStore(ABC):
    """議論グラフの永続化・検索を担う抽象基底クラス。"""

    # ── 書き込み ──────────────────────────────────────────────────────────────

    @abstractmethod
    def save_graph(self, graph: ArgumentGraph) -> None:
        """グラフ全体を保存する（冪等 — 同じIDは上書き）。"""

    @abstractmethod
    def upsert_node(self, node: ArgumentNode) -> None:
        """ノードを追加または更新する。"""

    @abstractmethod
    def upsert_edge(self, edge: ArgumentEdge) -> None:
        """エッジを追加または更新する。"""

    # ── 読み込み ──────────────────────────────────────────────────────────────

    @abstractmethod
    def get_node(self, node_id: str) -> ArgumentNode | None:
        """IDでノードを取得する。存在しない場合は None。"""

    @abstractmethod
    def get_graph(self, doc_id: str) -> ArgumentGraph:
        """doc_id に紐づくグラフ全体を返す。"""

    @abstractmethod
    def get_nodes_by_type(
        self,
        doc_id: str,
        node_type: NodeType,
    ) -> list[ArgumentNode]:
        """指定ドキュメント内の特定型ノードを全て返す。"""

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[ArgumentNode]:
        """隣接ノードを返す。

        Args:
            node_id: 起点ノードのID。
            edge_types: 辿るエッジ型名（None の場合は全て）。
            direction: "outgoing" | "incoming" | "both"
        """

    # ── 管理 ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None:
        """接続をクローズする。"""
