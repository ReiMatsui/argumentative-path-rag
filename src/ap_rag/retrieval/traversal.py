"""
型適応グラフ探索モジュール。

研究計画書 §4.2 のクエリ型ごとの探索戦略を実装する。
BFS（幅優先探索）で最大深度まで辿り、除外ノード型を自動フィルタリングする。
"""

from __future__ import annotations

import logging
from collections import deque

from ap_rag.graph.store import GraphStore
from ap_rag.models.graph import ArgumentNode
from ap_rag.models.taxonomy import TRAVERSAL_STRATEGIES, QueryType, TraversalStrategy

logger = logging.getLogger(__name__)


class GraphTraverser:
    """クエリ型に応じてグラフを探索し、関連ノードを収集する。

    Args:
        store: グラフストア。
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def traverse(
        self,
        entry_nodes: list[ArgumentNode],
        query_type: QueryType,
    ) -> list[ArgumentNode]:
        """エントリーノードから型適応的に探索し、関連ノードを返す。

        Args:
            entry_nodes: 探索の入口となるノードのリスト。
            query_type: クエリ型（探索戦略の選択に使う）。

        Returns:
            探索で収集したノードのリスト（エントリーノードを含む）。
            除外型のノードは含まれない。
        """
        strategy = TRAVERSAL_STRATEGIES[query_type]
        return self._bfs(entry_nodes, strategy)

    # ── BFS探索 ───────────────────────────────────────────────────────────────

    def _bfs(
        self,
        entry_nodes: list[ArgumentNode],
        strategy: TraversalStrategy,
    ) -> list[ArgumentNode]:
        """BFS でグラフを探索する。

        WHY型など「逆方向に辿る」戦略の場合、エッジの direction を
        "incoming"（エッジの向きを逆に辿る）に設定する。

        Args:
            entry_nodes: 探索の入口ノード。
            strategy: 探索戦略。

        Returns:
            収集したノードのリスト（訪問順）。
        """
        if strategy.max_depth == 0:
            # WHAT 型など探索不要の場合はエントリーのみ
            return [
                n for n in entry_nodes
                if n.node_type not in strategy.exclude_node_types
            ]

        visited: dict[str, ArgumentNode] = {}
        queue: deque[tuple[ArgumentNode, int]] = deque()

        exclude_types = set(strategy.exclude_node_types)
        edge_type_values = (
            [et.value for et in strategy.follow_edges]
            if strategy.follow_edges
            else None
        )

        # エントリーノードをキューに投入
        for node in entry_nodes:
            if node.node_type not in exclude_types and node.id not in visited:
                visited[node.id] = node
                queue.append((node, 0))

        while queue:
            current, depth = queue.popleft()

            if strategy.max_depth is not None and depth >= strategy.max_depth:
                continue

            # 探索方向: SUPPORTS / DERIVES / ASSUMES は「根拠 → 主張」方向なので
            # WHY型では incoming（逆向き）で辿ることで因果を遡る
            neighbors = self._store.get_neighbors(
                node_id=current.id,
                edge_types=edge_type_values,
                direction="incoming",
            )

            for neighbor in neighbors:
                if neighbor.id in visited:
                    continue
                if neighbor.node_type in exclude_types:
                    logger.debug(
                        "除外ノード型 %s をスキップ: %s",
                        neighbor.node_type.value,
                        neighbor.id,
                    )
                    continue
                visited[neighbor.id] = neighbor
                queue.append((neighbor, depth + 1))

        result = list(visited.values())
        logger.debug(
            "BFS完了: query_type=%s, entry=%d, 収集=%d ノード",
            strategy,
            len(entry_nodes),
            len(result),
        )
        return result
