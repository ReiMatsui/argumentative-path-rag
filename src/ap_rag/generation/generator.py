"""
LLM を使った回答生成モジュール。

役割付きコンテキストを受け取り、ユーザーへの回答を生成する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from ap_rag.openai_compat import reasoning_kwarg, sampling_kwargs
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
- Use the argumentative labels as internal hints to ground your reasoning, but \
  write a natural prose answer. Do NOT echo the role tags (do not output \
  "[CLAIM]", "[EVIDENCE]", "主張:", "根拠:", bullet scaffolds, etc.) in your answer.
- Match the granularity of the question:
    * Short factoid questions (e.g., "what datasets did they use?") → answer \
      in a single short sentence or a short comma-separated list.
    * WHY / HOW / reasoning questions → answer in 1-3 sentences that follow the \
      reasoning chain, still as natural prose.
- Answer in the same language as the query.
"""

_USER_TEMPLATE = """\
Context:
{context_text}

Question: {query}

Answer based only on the context above.
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
            **sampling_kwargs(self._model, temperature=0.1),
            **reasoning_kwarg(self._model),
        )
        answer = (response.choices[0].message.content or "").strip()
        logger.debug("回答生成完了: %d 文字", len(answer))
        return answer
