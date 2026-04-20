"""
評価指標モジュール。

研究計画書 §4.3「評価指標（複数並列）」に基づく実装:
  - EM / F1           : 回答の正答率
  - RAGAS faithfulness: 取得証拠が答えを支持する度合い
  - ハルシネーション率 : LLM-as-judge
  - 引用精度           : 文書内のどこを引用しているか
  - 回答の一貫性       : 同じ質問への回答の安定性
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any


# ── 評価サンプル ──────────────────────────────────────────────────────────────

@dataclass
class EvaluationSample:
    """1件の評価データ。

    Attributes:
        question: 質問文。
        ground_truth: 正解回答（文字列）。
        predicted_answer: システムの回答。
        retrieved_contexts: 検索で取得されたコンテキスト文字列のリスト。
        doc_id: 参照文書ID。
        query_type: クエリ型（WHY/WHAT/HOW/EVIDENCE/ASSUMPTION）。
        metadata: 任意の追加情報。
    """

    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_contexts: list[str]
    doc_id: str
    query_type: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """評価結果のサマリ。"""

    em: float                         # Exact Match
    f1: float                         # Token-level F1
    faithfulness: float | None        # RAGAS faithfulness（0-1）
    hallucination_rate: float | None  # ハルシネーション率（0-1、低いほど良い）
    citation_accuracy: float | None   # 引用精度
    answer_correctness: float | None  # LLM判定による回答正解度（0-1）
    num_samples: int
    per_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    raw_scores: list[dict[str, Any]] = field(default_factory=list)


# ── EM / F1 ───────────────────────────────────────────────────────────────────

def normalize_answer(text: str) -> str:
    """SQuAD スタイルの回答正規化。"""
    text = text.lower()
    # 句読点除去
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 日本語句読点除去
    text = re.sub(r"[。、！？「」『』【】（）]", " ", text)
    # 余分な空白を除去
    text = " ".join(text.split())
    return text


def compute_em(prediction: str, ground_truth: str) -> float:
    """Exact Match スコアを返す（0 または 1）。"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """トークンレベルの F1 スコアを返す（0.0〜1.0）。"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

_HALLUCINATION_JUDGE_PROMPT = """\
You are a factual accuracy judge. Given a question, a reference context, and a system answer, \
determine whether the answer contains hallucinations (information not grounded in the context).

Question: {question}

Reference context:
{context}

System answer:
{answer}

Does the answer contain any information that is NOT supported by the reference context?
Reply with ONLY one word: "YES" (hallucination present) or "NO" (fully grounded).
"""

_FAITHFULNESS_JUDGE_PROMPT = """\
You are evaluating retrieval faithfulness. Given a question, retrieved context passages, \
and a system answer, rate how faithfully the answer is grounded in the retrieved context.

Question: {question}

Retrieved context:
{context}

System answer:
{answer}

Rate the faithfulness on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context.
- 0.5: About half the claims are supported; the rest are inferred or unsupported.
- 0.0: The answer is entirely unsupported by the context.

Reply with ONLY a decimal number between 0.0 and 1.0.
"""

_ANSWER_CORRECTNESS_PROMPT = """\
You are evaluating whether a predicted answer correctly addresses a question.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted}

Judge whether the predicted answer conveys the same key information as the ground truth.
Be lenient with wording differences — focus on whether the core facts and meaning match.
A partial answer that covers the main point but omits minor details should score around 0.5-0.7.

Score from 0.0 to 1.0:
- 1.0: Fully correct — all key information matches the ground truth
- 0.7: Mostly correct — main point is right, minor details missing or slightly off
- 0.5: Partially correct — some key information matches, some is missing or wrong
- 0.2: Mostly incorrect — only superficially related to the ground truth
- 0.0: Incorrect — wrong answer or completely unrelated

Reply with ONLY a decimal number between 0.0 and 1.0.
"""


class LLMJudge:
    """LLM-as-judge によるハルシネーション率・忠実度の評価器。

    Args:
        client: openai.OpenAI インスタンス。
        model: 使用するモデル名。
    """

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    def is_hallucination(self, sample: EvaluationSample) -> bool:
        """回答にハルシネーションが含まれるか判定する。"""
        context = "\n".join(sample.retrieved_contexts)
        prompt = _HALLUCINATION_JUDGE_PROMPT.format(
            question=sample.question,
            context=context[:3000],  # トークン節約
            answer=sample.predicted_answer,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = (response.choices[0].message.content or "").strip().upper()
        return verdict.startswith("YES")

    def faithfulness_score(self, sample: EvaluationSample) -> float:
        """faithfulness スコアを返す（0.0〜1.0）。"""
        context = "\n".join(sample.retrieved_contexts)
        prompt = _FAITHFULNESS_JUDGE_PROMPT.format(
            question=sample.question,
            context=context[:3000],
            answer=sample.predicted_answer,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    def answer_correctness_score(self, sample: EvaluationSample) -> float:
        """LLM判定による回答正解度を返す（0.0〜1.0）。

        F1 がテキストの完全一致を要求するのに対し、こちらは意味的な正しさを評価する。
        言い換えや表現の違いを許容するため、グラフベース手法との相性が良い。

        ground_truth が空の場合は評価不能として 0.0 を返す。
        """
        if not sample.ground_truth.strip():
            return 0.0

        prompt = _ANSWER_CORRECTNESS_PROMPT.format(
            question=sample.question,
            ground_truth=sample.ground_truth[:1000],
            predicted=sample.predicted_answer[:1000],
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip()
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5


# ── アグリゲーション ──────────────────────────────────────────────────────────

def aggregate_results(
    samples: list[EvaluationSample],
    em_scores: list[float],
    f1_scores: list[float],
    faithfulness_scores: list[float] | None = None,
    hallucination_flags: list[bool] | None = None,
    answer_correctness_scores: list[float] | None = None,
) -> EvaluationResult:
    """個別スコアを集計して EvaluationResult を返す。"""
    n = len(samples)
    avg_em = sum(em_scores) / n if n > 0 else 0.0
    avg_f1 = sum(f1_scores) / n if n > 0 else 0.0

    avg_faith = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else None
    )
    hallucination_rate = (
        sum(1 for h in hallucination_flags if h) / len(hallucination_flags)
        if hallucination_flags
        else None
    )

    # クエリ型別の集計
    per_type: dict[str, dict[str, list[float]]] = {}
    for i, sample in enumerate(samples):
        qt = sample.query_type
        per_type.setdefault(qt, {"em": [], "f1": []})
        per_type[qt]["em"].append(em_scores[i])
        per_type[qt]["f1"].append(f1_scores[i])

    per_query_type = {
        qt: {
            "em": sum(v["em"]) / len(v["em"]),
            "f1": sum(v["f1"]) / len(v["f1"]),
            "count": len(v["em"]),
        }
        for qt, v in per_type.items()
    }

    avg_correctness = (
        sum(answer_correctness_scores) / len(answer_correctness_scores)
        if answer_correctness_scores
        else None
    )

    return EvaluationResult(
        em=avg_em,
        f1=avg_f1,
        faithfulness=avg_faith,
        hallucination_rate=hallucination_rate,
        citation_accuracy=None,  # MMDocRAG で計測（フェーズ2）
        answer_correctness=avg_correctness,
        num_samples=n,
        per_query_type=per_query_type,
    )
