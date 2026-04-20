"""
クエリ型分類モジュール。

入力クエリを WHY / WHAT / HOW / EVIDENCE / ASSUMPTION の5種に分類する。
軽量な gpt-4o-mini を使い、コストを最小化する。
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.models.taxonomy import QueryType
from ap_rag.openai_compat import max_tokens_kwarg, reasoning_kwarg, sampling_kwargs

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a query classifier for a Retrieval-Augmented Generation system.
Classify the user's query into exactly one of the following types:

- WHY: Asks for reasons, causes, or explanations. (e.g., "Why did X happen?")
- WHAT: Asks for facts, definitions, or values. (e.g., "What is X?" / "How much is X?")
- HOW: Asks for procedures, methods, or processes. (e.g., "How do I do X?")
- EVIDENCE: Asks for supporting evidence or proof. (e.g., "What is the evidence for X?")
- ASSUMPTION: Asks for underlying premises or prerequisites. (e.g., "What assumptions does X make?")

Reply with ONLY the type name (WHY, WHAT, HOW, EVIDENCE, or ASSUMPTION). No explanation.
"""


class QueryClassifier:
    """クエリを QueryType に分類する。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名（コスト重視なら mini 推奨）。
    """

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def classify(self, query: str) -> QueryType:
        """クエリを分類する。

        不明な場合のデフォルトは WHAT。

        Args:
            query: ユーザー入力テキスト。

        Returns:
            QueryType。
        """
        raw = self._call_llm(query)
        return self._parse(raw)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def _call_llm(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            **sampling_kwargs(self._model, temperature=0.0),
            **max_tokens_kwarg(self._model, 10),
            **reasoning_kwarg(self._model),
        )
        return (response.choices[0].message.content or "").strip().upper()

    @staticmethod
    def _parse(raw: str) -> QueryType:
        """LLM の出力を QueryType に変換する。マッチしなければ WHAT を返す。"""
        try:
            return QueryType(raw)
        except ValueError:
            # 部分一致でフォールバック
            for qt in QueryType:
                if qt.value in raw:
                    return qt
            logger.warning("クエリ型の解析失敗: %r → WHAT にフォールバック", raw)
            return QueryType.WHAT
