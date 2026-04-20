"""
議論グラフのコアデータモデル。

ArgumentNode / ArgumentEdge は不変値オブジェクトとして設計し、
グラフ操作はすべて ArgumentGraph を通じて行う。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ap_rag.models.taxonomy import EdgeType, NodeType


# ── ノード ────────────────────────────────────────────────────────────────────

class ArgumentNode(BaseModel):
    """議論グラフの1ノードを表す不変モデル。

    Attributes:
        id: UUID文字列（デフォルトで自動生成）。
        node_type: タクソノミー上のノード種別。
        text: ノードが表すテキスト内容。
        source_doc_id: 元文書の識別子。
        source_chunk_idx: 元文書内のチャンクインデックス。
        embedding: 埋め込みベクトル（入口選定時に付与）。
        metadata: 任意の追加情報。
        created_at: 生成タイムスタンプ（UTC）。
    """

    model_config = {"frozen": True}  # 不変（値オブジェクト）

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType
    text: str
    source_doc_id: str
    source_chunk_idx: int = Field(ge=0)
    embedding: list[float] | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text は空文字列にできません")
        return v.strip()


# ── エッジ ────────────────────────────────────────────────────────────────────

class ArgumentEdge(BaseModel):
    """議論グラフの有向エッジを表す不変モデル。

    Attributes:
        id: UUID文字列。
        edge_type: エッジ種別。
        source_id: 始点ノードのID。
        target_id: 終点ノードのID。
        confidence: LLM抽出時の確信度（0.0〜1.0）。
        metadata: 任意の追加情報。
    """

    model_config = {"frozen": True}

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    edge_type: EdgeType
    source_id: str
    target_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def source_and_target_must_differ(self) -> ArgumentEdge:
        if self.source_id == self.target_id:
            raise ValueError("自己ループエッジは不正です（source_id == target_id）")
        return self


# ── グラフ ────────────────────────────────────────────────────────────────────

class ArgumentGraph(BaseModel):
    """1文書から構築された議論グラフ全体。

    Attributes:
        doc_id: 元文書の識別子。
        nodes: ノードID → ArgumentNode のマッピング。
        edges: エッジID → ArgumentEdge のマッピング。
    """

    doc_id: str
    nodes: dict[str, ArgumentNode] = Field(default_factory=dict)
    edges: dict[str, ArgumentEdge] = Field(default_factory=dict)

    # ── 変更操作 ──────────────────────────────────────────────────────────────

    def add_node(self, node: ArgumentNode) -> None:
        """ノードを追加する。ID重複時は上書きせず ValueError を送出する。"""
        if node.id in self.nodes:
            raise ValueError(f"ノードID {node.id!r} は既に存在します")
        self.nodes[node.id] = node

    def add_edge(self, edge: ArgumentEdge) -> None:
        """エッジを追加する。端点ノードが存在しない場合は ValueError を送出する。"""
        if edge.source_id not in self.nodes:
            raise ValueError(f"始点ノード {edge.source_id!r} が存在しません")
        if edge.target_id not in self.nodes:
            raise ValueError(f"終点ノード {edge.target_id!r} が存在しません")
        if edge.id in self.edges:
            raise ValueError(f"エッジID {edge.id!r} は既に存在します")
        self.edges[edge.id] = edge

    # ── 参照操作 ──────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> ArgumentNode:
        """ノードを取得する。存在しない場合は KeyError を送出する。"""
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"ノードID {node_id!r} が見つかりません") from None

    def outgoing_edges(self, node_id: str) -> list[ArgumentEdge]:
        """指定ノードから出るエッジを返す。"""
        return [e for e in self.edges.values() if e.source_id == node_id]

    def incoming_edges(self, node_id: str) -> list[ArgumentEdge]:
        """指定ノードに入るエッジを返す。"""
        return [e for e in self.edges.values() if e.target_id == node_id]

    def nodes_by_type(self, node_type: NodeType) -> list[ArgumentNode]:
        """指定型のノード一覧を返す。"""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    # ── 統計 ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """ノード・エッジの統計情報を返す。"""
        node_counts = {nt.value: 0 for nt in NodeType}
        for n in self.nodes.values():
            node_counts[n.node_type.value] += 1

        edge_counts = {et.value: 0 for et in EdgeType}
        for e in self.edges.values():
            edge_counts[e.edge_type.value] += 1

        return {
            "doc_id": self.doc_id,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }
