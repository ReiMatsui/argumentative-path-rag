"""
Neo4j を使った議論グラフストアの実装。

スキーマ設計:
  (:ArgumentNode {id, node_type, text, source_doc_id, source_chunk_idx, ...})
  -[:SUPPORTS|CONTRADICTS|DERIVES|ASSUMES|ILLUSTRATES|CONTRASTS {id, confidence}]->
  (:ArgumentNode {...})
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Session

from ap_rag.graph.store import GraphStore
from ap_rag.models.graph import ArgumentEdge, ArgumentGraph, ArgumentNode
from ap_rag.models.taxonomy import EdgeType, NodeType

logger = logging.getLogger(__name__)

# ── Cypher クエリ定義 ─────────────────────────────────────────────────────────

_UPSERT_NODE = """
MERGE (n:ArgumentNode {id: $id})
SET n.node_type        = $node_type,
    n.text             = $text,
    n.source_doc_id    = $source_doc_id,
    n.source_chunk_idx = $source_chunk_idx,
    n.metadata         = $metadata
"""

_UPSERT_EDGE = """
MATCH (src:ArgumentNode {id: $source_id})
MATCH (tgt:ArgumentNode {id: $target_id})
MERGE (src)-[r:{edge_type} {{id: $id}}]->(tgt)
SET r.confidence = $confidence,
    r.metadata   = $metadata
"""

_GET_NODE = """
MATCH (n:ArgumentNode {id: $id})
RETURN n
"""

_GET_NODES_BY_TYPE = """
MATCH (n:ArgumentNode {source_doc_id: $doc_id, node_type: $node_type})
RETURN n
"""

_GET_ALL_NODES = """
MATCH (n:ArgumentNode {source_doc_id: $doc_id})
RETURN n
"""

_GET_ALL_EDGES = """
MATCH (src:ArgumentNode {source_doc_id: $doc_id})-[r]->(tgt:ArgumentNode)
RETURN r, src.id AS source_id, tgt.id AS target_id, type(r) AS edge_type
"""

_GET_NEIGHBORS_OUTGOING = """
MATCH (n:ArgumentNode {id: $node_id})-[r]->(neighbor:ArgumentNode)
WHERE $edge_types IS NULL OR type(r) IN $edge_types
RETURN neighbor, type(r) AS edge_type
"""

_GET_NEIGHBORS_INCOMING = """
MATCH (neighbor:ArgumentNode)-[r]->(n:ArgumentNode {id: $node_id})
WHERE $edge_types IS NULL OR type(r) IN $edge_types
RETURN neighbor, type(r) AS edge_type
"""

_CREATE_INDEXES = [
    "CREATE INDEX node_id IF NOT EXISTS FOR (n:ArgumentNode) ON (n.id)",
    "CREATE INDEX node_doc IF NOT EXISTS FOR (n:ArgumentNode) ON (n.source_doc_id)",
    "CREATE INDEX node_type IF NOT EXISTS FOR (n:ArgumentNode) ON (n.node_type)",
]


class Neo4jGraphStore(GraphStore):
    """Neo4j を使った議論グラフストア。

    Args:
        uri: Neo4j Bolt URI（例: "bolt://localhost:7687"）。
        user: ユーザー名。
        password: パスワード。
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_indexes()

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        with self._driver.session() as session:
            yield session

    def _ensure_indexes(self) -> None:
        with self._session() as session:
            for cypher in _CREATE_INDEXES:
                session.run(cypher)

    # ── 書き込み ──────────────────────────────────────────────────────────────

    def save_graph(self, graph: ArgumentGraph) -> None:
        """グラフ全体を Neo4j に保存する。"""
        for node in graph.nodes.values():
            self.upsert_node(node)
        for edge in graph.edges.values():
            self.upsert_edge(edge)
        logger.info(
            "グラフ保存完了: doc_id=%s, nodes=%d, edges=%d",
            graph.doc_id,
            len(graph.nodes),
            len(graph.edges),
        )

    def upsert_node(self, node: ArgumentNode) -> None:
        with self._session() as session:
            session.run(
                _UPSERT_NODE,
                id=node.id,
                node_type=node.node_type.value,
                text=node.text,
                source_doc_id=node.source_doc_id,
                source_chunk_idx=node.source_chunk_idx,
                metadata=json.dumps(node.metadata, ensure_ascii=False),
            )

    def upsert_edge(self, edge: ArgumentEdge) -> None:
        cypher = _UPSERT_EDGE.format(edge_type=edge.edge_type.value)
        with self._session() as session:
            session.run(
                cypher,
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                confidence=edge.confidence,
                metadata=json.dumps(edge.metadata, ensure_ascii=False),
            )

    # ── 読み込み ──────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> ArgumentNode | None:
        with self._session() as session:
            result = session.run(_GET_NODE, id=node_id)
            record = result.single()
            if record is None:
                return None
            return self._record_to_node(record["n"])

    def get_graph(self, doc_id: str) -> ArgumentGraph:
        graph = ArgumentGraph(doc_id=doc_id)
        with self._session() as session:
            # ノード
            for record in session.run(_GET_ALL_NODES, doc_id=doc_id):
                node = self._record_to_node(record["n"])
                graph.nodes[node.id] = node
            # エッジ
            for record in session.run(_GET_ALL_EDGES, doc_id=doc_id):
                edge = self._record_to_edge(record)
                if edge is not None:
                    graph.edges[edge.id] = edge
        return graph

    def get_nodes_by_type(
        self,
        doc_id: str,
        node_type: NodeType,
    ) -> list[ArgumentNode]:
        with self._session() as session:
            result = session.run(
                _GET_NODES_BY_TYPE,
                doc_id=doc_id,
                node_type=node_type.value,
            )
            return [self._record_to_node(r["n"]) for r in result]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "outgoing",
    ) -> list[ArgumentNode]:
        if direction == "outgoing":
            cypher = _GET_NEIGHBORS_OUTGOING
        elif direction == "incoming":
            cypher = _GET_NEIGHBORS_INCOMING
        else:
            raise ValueError(f"direction は 'outgoing' か 'incoming' のみ有効: {direction!r}")

        with self._session() as session:
            result = session.run(
                cypher,
                node_id=node_id,
                edge_types=edge_types,
            )
            return [self._record_to_node(r["neighbor"]) for r in result]

    # ── 管理 ──────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    # ── 変換ヘルパー ──────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_node(neo4j_node: Any) -> ArgumentNode:
        props = dict(neo4j_node)
        metadata = props.get("metadata", "{}")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return ArgumentNode(
            id=props["id"],
            node_type=NodeType(props["node_type"]),
            text=props["text"],
            source_doc_id=props["source_doc_id"],
            source_chunk_idx=int(props["source_chunk_idx"]),
            metadata=metadata,
        )

    @staticmethod
    def _record_to_edge(record: Any) -> ArgumentEdge | None:
        try:
            rel = record["r"]
            props = dict(rel)
            return ArgumentEdge(
                id=props["id"],
                edge_type=EdgeType(record["edge_type"]),
                source_id=record["source_id"],
                target_id=record["target_id"],
                confidence=float(props.get("confidence", 1.0)),
            )
        except (KeyError, ValueError) as e:
            logger.warning("エッジ変換失敗: %s", e)
            return None
