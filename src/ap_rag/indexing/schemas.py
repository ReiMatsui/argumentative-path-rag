"""
インデックスパイプライン向け LLM 出力スキーマ定義。

OpenAI Structured Outputs (`client.beta.chat.completions.parse`) に渡す
Pydantic モデルをここに集約する。NodeType / EdgeType の enum を
フィールドに直接使うことで、API レベルで不正値を排除できる。

モデルを変更したい場合は .env の OPENAI_CLASSIFIER_MODEL を書き換えるだけでよい:
    OPENAI_CLASSIFIER_MODEL=gpt-5        # 将来の移行例
    OPENAI_CLASSIFIER_MODEL=gpt-4o-mini  # コスト重視の場合
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ap_rag.models.taxonomy import EdgeType, NodeType


# ── ノード分類スキーマ ─────────────────────────────────────────────────────────

class NodeItem(BaseModel):
    """LLM が1件のノードとして返す出力単位。"""

    node_type: NodeType = Field(
        description=(
            "Argumentative role. Must be exactly one of: "
            "CLAIM, EVIDENCE, ASSUMPTION, CONCLUSION, CAVEAT, CONTRAST, DEFINITION."
        )
    )
    text: str = Field(
        description=(
            "A short normalized label for display (may be a paraphrase). "
            "Keep it under ~200 characters. The lossless original is in source_span."
        )
    )
    source_span: str = Field(
        default="",
        description=(
            "Verbatim substring of the input passage that this node covers. "
            "MUST be a character-for-character copy from the passage — do not "
            "rewrite, summarize, translate, or normalize. Preserve numbers, "
            "named entities, citations, and punctuation exactly. "
            "If absolutely no verbatim substring applies (e.g. the unit is a "
            "synthesis of multiple sentences), return an empty string."
        ),
    )


class NodeClassificationOutput(BaseModel):
    """NodeClassifier が受け取る LLM 出力全体。"""

    nodes: list[NodeItem] = Field(
        description="All argumentative units extracted from the passage.",
        default_factory=list,
    )


# ── エッジ抽出スキーマ ─────────────────────────────────────────────────────────

class EdgeItem(BaseModel):
    """LLM が1件のエッジとして返す出力単位。"""

    source_idx: int = Field(
        description="0-based index of the source node in the provided nodes list."
    )
    target_idx: int = Field(
        description="0-based index of the target node in the provided nodes list."
    )
    edge_type: EdgeType = Field(
        description=(
            "Directed relationship type. Must be exactly one of: "
            "SUPPORTS, CONTRADICTS, DERIVES, ASSUMES, ILLUSTRATES, CONTRASTS."
        )
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class EdgeExtractionOutput(BaseModel):
    """EdgeExtractor が受け取る LLM 出力全体。"""

    edges: list[EdgeItem] = Field(
        description="All directed argumentative relationships identified between the nodes.",
        default_factory=list,
    )
