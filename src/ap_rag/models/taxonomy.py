"""
議論グラフのタクソノミー定義。

研究計画書 §2.3 に基づく 7ノード × 6エッジ の型体系と、
§4.2 に基づくクエリ型の定義をここに集約する。
"""

from __future__ import annotations

from enum import Enum


# ── ノード型（7種） ────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """議論グラフのノード種別。

    各値は Neo4j ラベル・プロンプト文字列としてそのまま使用される。
    """

    CLAIM = "CLAIM"
    """中心的な主張・命題。WHY/WHAT クエリの主要入口となる。"""

    EVIDENCE = "EVIDENCE"
    """主張を支持する実証的根拠（数値・実験結果・引用など）。"""

    ASSUMPTION = "ASSUMPTION"
    """主張が暗黙的に前提としている条件・背景知識。"""

    CONCLUSION = "CONCLUSION"
    """複数の根拠・推論から導かれる結論。WHY クエリの入口にもなる。"""

    CAVEAT = "CAVEAT"
    """主張の適用範囲を限定する注意・例外・留保。"""

    CONTRAST = "CONTRAST"
    """主張に対比・反例を提示する記述。WHY クエリでは原則除外される。"""

    DEFINITION = "DEFINITION"
    """概念・用語の定義。"""


# ── エッジ型（6種） ────────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    """議論グラフのエッジ種別。

    各値は Neo4j リレーションシップ型としてそのまま使用される。
    """

    SUPPORTS = "SUPPORTS"
    """src が tgt の主張を支持する（EVIDENCE → CLAIM など）。"""

    CONTRADICTS = "CONTRADICTS"
    """src が tgt の主張に反する（CONTRAST → CLAIM など）。"""

    DERIVES = "DERIVES"
    """src から tgt が論理的に導出される（EVIDENCE → CONCLUSION など）。"""

    ASSUMES = "ASSUMES"
    """tgt は src の成立を前提とする（CLAIM → ASSUMPTION の逆読み）。"""

    ILLUSTRATES = "ILLUSTRATES"
    """src が tgt を具体例・図表で示す（視覚ノード → CLAIM など）。"""

    CONTRASTS = "CONTRASTS"
    """src が tgt と対比関係にある（CONTRAST → CLAIM など）。"""


# ── クエリ型（5種） ────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    """入力クエリの意図分類。

    研究計画書 §4.2 のクエリ型ごとの探索戦略に対応。
    """

    WHY = "WHY"
    """「なぜXが起きたか」― 因果・根拠を問う。"""

    WHAT = "WHAT"
    """「Xは何か / いくらか」― 事実・定義を問う。"""

    HOW = "HOW"
    """「どうやってXをするか」― 手順・方法を問う。"""

    EVIDENCE = "EVIDENCE"
    """「Xの根拠は何か」― 支持証拠を問う。"""

    ASSUMPTION = "ASSUMPTION"
    """「Xの前提は何か」― 暗黙的前提を問う。"""


# ── 探索戦略の静的定義 ────────────────────────────────────────────────────────

class TraversalStrategy:
    """クエリ型ごとの探索パラメータ（研究計画書 §4.2 テーブルを構造化）。

    Attributes:
        entry_node_types: グラフ探索の入口として優先するノード型。
        follow_edges: 辿るエッジ型のリスト。
        exclude_node_types: 取得結果から除外するノード型。
        max_depth: グラフ探索の最大深度（None = 制限なし）。
        edge_directions: エッジ型ごとの探索方向。
            "incoming" = エッジの向きを逆に辿る（SUPPORTS: EVIDENCE→CLAIM なら CLAIM から辿ると EVIDENCE が見つかる）。
            "outgoing" = エッジの向きのまま辿る（ASSUMES: CLAIM→ASSUMPTION なら CLAIM から辿ると ASSUMPTION が見つかる）。
            指定のないエッジ型はデフォルト値（"incoming"）を使用。
    """

    _DEFAULT_DIRECTION = "incoming"

    def __init__(
        self,
        entry_node_types: list[NodeType],
        follow_edges: list[EdgeType],
        exclude_node_types: list[NodeType],
        max_depth: int | None = 3,
        edge_directions: dict[EdgeType, str] | None = None,
    ) -> None:
        self.entry_node_types = entry_node_types
        self.follow_edges = follow_edges
        self.exclude_node_types = exclude_node_types
        self.max_depth = max_depth
        self.edge_directions: dict[EdgeType, str] = edge_directions or {}

    def direction_for(self, edge_type: EdgeType) -> str:
        """指定エッジ型の探索方向を返す。未指定の場合はデフォルト値。"""
        return self.edge_directions.get(edge_type, self._DEFAULT_DIRECTION)

    def edges_by_direction(self) -> dict[str, list[str]]:
        """エッジ型を方向ごとにグループ化して返す。

        Returns:
            {"incoming": ["SUPPORTS", ...], "outgoing": ["ASSUMES", ...]}
        """
        groups: dict[str, list[str]] = {"incoming": [], "outgoing": []}
        for et in self.follow_edges:
            direction = self.direction_for(et)
            groups[direction].append(et.value)
        return groups


# 研究計画書 §4.2 テーブルを忠実に実装
#
# エッジ方向の根拠:
#   SUPPORTS   : EVIDENCE  → CLAIM       incoming（CLAIMに入ってくるEVIDENCEを探す）
#   DERIVES    : EVIDENCE  → CONCLUSION  incoming（CONCLUSIONに入ってくる根拠を探す）
#   ASSUMES    : CLAIM     → ASSUMPTION  outgoing（CLAIMから出ていくASSUMPTIONを探す）
#   ILLUSTRATES: visual    → CLAIM       incoming
#   CONTRASTS  : CONTRAST  → CLAIM       incoming
TRAVERSAL_STRATEGIES: dict[QueryType, TraversalStrategy] = {
    QueryType.WHY: TraversalStrategy(
        entry_node_types=[NodeType.CLAIM, NodeType.CONCLUSION],
        follow_edges=[EdgeType.SUPPORTS, EdgeType.DERIVES, EdgeType.ASSUMES],
        exclude_node_types=[NodeType.CONTRAST],
        max_depth=3,
        edge_directions={
            EdgeType.SUPPORTS: "incoming",
            EdgeType.DERIVES:  "incoming",
            EdgeType.ASSUMES:  "outgoing",   # CLAIM → ASSUMPTION
        },
    ),
    QueryType.WHAT: TraversalStrategy(
        entry_node_types=[NodeType.CLAIM, NodeType.DEFINITION],
        follow_edges=[],  # 入口ノードのみ（探索なし）
        exclude_node_types=[],
        max_depth=0,
    ),
    QueryType.HOW: TraversalStrategy(
        entry_node_types=[NodeType.CLAIM, NodeType.CONCLUSION],
        follow_edges=[EdgeType.ILLUSTRATES, EdgeType.DERIVES],
        exclude_node_types=[],
        max_depth=3,
        edge_directions={
            EdgeType.ILLUSTRATES: "incoming",
            EdgeType.DERIVES:     "incoming",
        },
    ),
    QueryType.EVIDENCE: TraversalStrategy(
        entry_node_types=[NodeType.CLAIM],
        follow_edges=[EdgeType.SUPPORTS],
        exclude_node_types=[NodeType.CONTRAST, NodeType.ASSUMPTION],
        max_depth=2,
        edge_directions={
            EdgeType.SUPPORTS: "incoming",
        },
    ),
    QueryType.ASSUMPTION: TraversalStrategy(
        entry_node_types=[NodeType.CLAIM, NodeType.CONCLUSION],
        follow_edges=[EdgeType.ASSUMES],
        exclude_node_types=[],
        max_depth=2,
        edge_directions={
            EdgeType.ASSUMES: "outgoing",    # CLAIM → ASSUMPTION
        },
    ),
}
