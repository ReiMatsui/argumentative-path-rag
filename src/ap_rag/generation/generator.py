"""
LLM を使った回答生成モジュール。

役割付きコンテキストを受け取り、ユーザーへの回答を生成する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.retrieval.context_builder import RetrievalContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. You will be given:
1. A user's query.
2. Structured context retrieved from an argument graph, with each piece labeled \
   by its argumentative role (e.g., [CLAIM], [EVIDENCE], [ASSUMPTION]).

Your task:
- Answer the query based ONLY on the provided context.
- If the answer cannot be determined from the context, say so explicitly.
- Distinguish between what is claimed, what is supported by evidence, and what \
  is assumed — use this structure in your answer when relevant.
- Be concise but complete. Cite the relevant parts of the context in your answer.
- Answer in the same language as the query.
"""

_USER_TEMPLATE = """\
{context_text}

質問: {query}

上記の参照情報に基づいて回答してください。
"""


@dataclass
class GenerationResult:
    """回答生成の結果。"""

    query: str
    answer: str
    retrieval_context: RetrievalContext
    model: str


class AnswerGenerator:
    """コンテキストを元に LLM で回答を生成する。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名。
    """

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def generate(self, context: RetrievalContext) -> GenerationResult:
        """コンテキストから回答を生成する。

        Args:
            context: ContextBuilder が構築した RetrievalContext。

        Returns:
            GenerationResult。
        """
        answer = self._call_llm(context)
        return GenerationResult(
            query=context.query,
            answer=answer,
            retrieval_context=context,
            model=self._model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, context: RetrievalContext) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        context_text=context.context_text,
                        query=context.query,
                    ),
                },
            ],
            temperature=0.1,
        )
        answer = (response.choices[0].message.content or "").strip()
        logger.debug("回答生成完了: %d 文字", len(answer))
        return answer
