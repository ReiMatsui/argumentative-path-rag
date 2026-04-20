"""
アプリケーション設定管理。
pydantic-settings が .env ファイルを自動で読み込む。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API キー")
    openai_classifier_model: str = Field(
        default="gpt-4o",
        description="ノード分類・エッジ抽出に使うモデル",
    )
    openai_generator_model: str = Field(
        default="gpt-4o-mini",
        description="クエリ型分類・回答生成に使うモデル",
    )
    llm_max_retries: int = Field(default=3, description="LLM呼び出しの最大リトライ数")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="intfloat/e5-mistral-7b-instruct",
        description="入口選定に使うEmbeddingモデル",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Embeddingモデルの実行デバイス",
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="ap_rag_dev")

    # ── General ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """シングルトンとして設定を返す（テスト時は cache_clear() でリセット可）。"""
    return Settings()  # type: ignore[call-arg]
