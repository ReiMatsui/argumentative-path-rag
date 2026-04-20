"""
NetworkX を使ったインメモリ議論グラフストア。

Neo4j の代替として、テスト・プロトタイピング・デモ用に使用する。
GraphStore インターフェースを完全に実装するため、
pipeline.py を一切変えず差し替え可能。
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

from ap_rag.graph.store import GraphStore
from ap_rag.models.graph import ArgumentEdge, ArgumentGraph, ArgumentNode
from ap_rag.models.taxonomy import NodeType

logger = logging.getLogger(__name__)


class NetworkXGraphStore(GraphStore):
    """NetworkX ベースのインメモリグラフストア。

    プロセスが終了するとデータは消える。
    """

    def __init__(self) -> None:
        # doc_id → nx.MultiDiGraph
        self._graphs: dict[str, nx.MultiDiGraph] = defaultdict(nx.MultiDiGraph)
        # node_id → ArgumentNode（全doc共通の索引）
        self._nodes: dict[str, ArgumentNode] = {}

    # ── 書き込み ──────────────────────────────────────────────────────────────

    def save_graph(self, graph: ArgumentGraph) -> None:
        for node in graph.nodes.values():
            self.upsert_node(node)
        for edge in graph.edges.values():
            self.upsert_edge(edge)
        logger.info(
            "グラフ保存完了（NetworkX）: doc_id=%s, nodes=%d, edges=%d",
            graph.doc_id,
            len(graph.nodes),
            len(graph.edges),
        )

    def upsert_node(self, node: ArgumentNode) -> None:
        self._nodes[node.id] = node
        g = self._graphs[node.source_doc_id]
        g.add_node(
            node.id,
            node_type=node.node_type.value,
            text=node.text,
            source_chunk_idx=node.source_chunk_idx,
        )

    def upsert_edge(self, edge: ArgumentEdge) -> None:
        # ノードが属するdoc_idを解決する
        src_node = self._nodes.get(edge.source_id)
        if src_node is None:
            logger.warning("始点ノードが見つかりません: %s", edge.source_id)
            return
        g = self._graphs[src_node.source_doc_id]
        g.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            edge_type=edge.edge_type.value,
            confidence=edge.confidence,
        )

    # ── 読み込み ──────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> ArgumentNode | None:
        return self._nodes.get(node_id)

    def get_graph(self, doc_id: str) -> ArgumentGraph:
        graph = ArgumentGraph(doc_id=doc_id)
        g = self._graphs.get(doc_id)
        if g is None:
            return graph

        for node_id in g.nodes:
            node = self._nodes.get(node_id)
            if node:
                graph.nodes[node_id] = node

        for src, tgt, key, data in g.edges(keys=True, data=True):
            try:
                from ap_rag.models.taxonomy import EdgeType
                edge = ArgumentEdge(
                    id=key,
                    edge_type=EdgeType(data["edge_type"]),
                    source_id=src,
                    target_id=tgt,
                    confidence=data.get("confidence", 1.0),
                )
                graph.edges[key] = edge
            except Exception as e:
                logger.warning("エッジ変換失敗: %s", e)

        return graph

    def get_nodes_by_type(self, doc_id: str, node_type: NodeType) -> list[ArgumentNode]:
        g = self._graphs.get(doc_id)
        if g is None:
            return []
        return [
            self._nodes[nid]
            for nid in g.nodes
            if nid in self._nodes
            and self._nodes[nid].node_type == node_type
        ]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[ArgumentNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        g = self._graphs.get(node.source_doc_id)
        if g is None:
            return []

        if direction == "outgoing":
            edges_iter = g.out_edges(node_id, keys=True, data=True)
            neighbor_ids = [
                tgt for _, tgt, key, data in edges_iter
                if edge_types is None or data.get("edge_type") in edge_types
            ]
        elif direction == "incoming":
            edges_iter = g.in_edges(node_id, keys=True, data=True)
            neighbor_ids = [
                src for src, _, key, data in edges_iter
                if edge_types is None or data.get("edge_type") in edge_types
            ]
        else:
            raise ValueError(f"direction は 'outgoing' か 'incoming' のみ有効: {direction!r}")

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    # ── 管理 ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """インメモリなので何もしない。"""
        pass

    def stats(self) -> dict:
        return {
            doc_id: {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
            for doc_id, g in self._graphs.items()
        }
