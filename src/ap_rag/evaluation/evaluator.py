"""
評価実行器。

ArgumentativeRAGPipeline と BaselineRAG を受け取り、
EvaluationSample のリストに対して複数指標を並列計算する。

研究計画書 §4.3「複数指標を並列に使って多面的に立証する」設計。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from ap_rag.evaluation.metrics import (
    EvaluationResult,
    EvaluationSample,
    LLMJudge,
    aggregate_results,
    compute_em,
    compute_f1,
)

logger = logging.getLogger(__name__)


# ── RAGシステムの共通インターフェース ─────────────────────────────────────────

class RAGSystem(Protocol):
    """評価対象のRAGシステムが実装すべきインターフェース。

    ArgumentativeRAGPipeline と BaselineRAG の両方がこれを満たす。
    """

    def query(self, question: str, doc_id: str) -> Any:
        """質問に対する回答を生成する。result.answer と result.retrieval_context.nodes を持つ。"""
        ...


# ── 評価実行器 ────────────────────────────────────────────────────────────────

class Evaluator:
    """RAGシステムを評価サンプルに対して実行し、結果を集計する。

    Args:
        system: 評価対象のRAGシステム。
        judge: LLM-as-judge（None の場合は faithfulness・ハルシネーション率をスキップ）。
        show_progress: プログレスバーを表示するか。
    """

    def __init__(
        self,
        system: RAGSystem,
        judge: LLMJudge | None = None,
        show_progress: bool = True,
    ) -> None:
        self._system = system
        self._judge = judge
        self._show_progress = show_progress

    def evaluate(
        self,
        samples: list[EvaluationSample],
        use_judge: bool = True,
    ) -> EvaluationResult:
        """サンプルリスト全体を評価して EvaluationResult を返す。

        Args:
            samples: 評価対象サンプルのリスト。
            use_judge: LLM-as-judge を使うか（コスト節約のため False にできる）。

        Returns:
            EvaluationResult — 集計済みの評価結果。
        """
        em_scores: list[float] = []
        f1_scores: list[float] = []
        faithfulness_scores: list[float] = []
        hallucination_flags: list[bool] = []
        answer_correctness_scores: list[float] = []
        completed_samples: list[EvaluationSample] = []

        run_judge = use_judge and self._judge is not None

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            disable=not self._show_progress,
        ) as progress:
            task = progress.add_task("評価中...", total=len(samples))

            for sample in samples:
                result_sample = self._evaluate_one(
                    sample,
                    run_judge,
                    faithfulness_scores,
                    hallucination_flags,
                    answer_correctness_scores,
                )
                em_scores.append(compute_em(result_sample.predicted_answer, sample.ground_truth))
                f1_scores.append(compute_f1(result_sample.predicted_answer, sample.ground_truth))
                completed_samples.append(result_sample)
                progress.advance(task)

        return aggregate_results(
            samples=completed_samples,
            em_scores=em_scores,
            f1_scores=f1_scores,
            faithfulness_scores=faithfulness_scores if faithfulness_scores else None,
            hallucination_flags=hallucination_flags if hallucination_flags else None,
            answer_correctness_scores=answer_correctness_scores if answer_correctness_scores else None,
        )

    def _evaluate_one(
        self,
        sample: EvaluationSample,
        run_judge: bool,
        faithfulness_scores: list[float],
        hallucination_flags: list[bool],
        answer_correctness_scores: list[float],
    ) -> EvaluationSample:
        """1サンプルを評価し、predicted_answer と retrieved_contexts を埋めたサンプルを返す。"""
        try:
            result = self._system.query(sample.question, sample.doc_id)
            predicted_answer = result.answer
            # コンテキストテキストを retrieved_contexts に格納
            retrieved_contexts = [
                node.text
                for node in getattr(result.retrieval_context, "nodes", [])
            ]
        except Exception as e:
            logger.warning("クエリ実行失敗: %s — question=%s", e, sample.question[:50])
            predicted_answer = ""
            retrieved_contexts = []

        # サンプルに結果を書き込む（dataclassなので新しいインスタンスを作る）
        from dataclasses import replace
        completed = replace(
            sample,
            predicted_answer=predicted_answer,
            retrieved_contexts=retrieved_contexts,
        )

        # LLM judge
        if run_judge and self._judge is not None:
            try:
                faithfulness_scores.append(self._judge.faithfulness_score(completed))
                hallucination_flags.append(self._judge.is_hallucination(completed))
                answer_correctness_scores.append(
                    self._judge.answer_correctness_score(completed)
                )
            except Exception as e:
                logger.warning("LLM judge 失敗: %s", e)

        return completed


# ── 比較実行 ──────────────────────────────────────────────────────────────────

class ComparisonRunner:
    """複数のRAGシステムを同一サンプルで比較評価する。

    研究計画書 §4.3「比較ベースライン」に対応。
    """

    def __init__(
        self,
        systems: dict[str, RAGSystem],
        judge: LLMJudge | None = None,
    ) -> None:
        self._systems = systems
        self._judge = judge

    def run(
        self,
        samples: list[EvaluationSample],
        use_judge: bool = False,
    ) -> dict[str, EvaluationResult]:
        """全システムを評価して {system_name: EvaluationResult} を返す。"""
        results: dict[str, EvaluationResult] = {}
        for name, system in self._systems.items():
            logger.info("評価中: %s", name)
            evaluator = Evaluator(system, judge=self._judge, show_progress=True)
            results[name] = evaluator.evaluate(samples, use_judge=use_judge)
        return results

    @staticmethod
    def print_comparison(results: dict[str, EvaluationResult]) -> None:
        """比較結果をターミナルに表示する。"""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="システム比較結果", show_header=True, header_style="bold magenta")
        table.add_column("システム", style="bold")
        table.add_column("EM", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Faithfulness", justify="right")
        table.add_column("Hallucination↓", justify="right")
        table.add_column("N", justify="right")

        for name, result in results.items():
            table.add_row(
                name,
                f"{result.em:.3f}",
                f"{result.f1:.3f}",
                f"{result.faithfulness:.3f}" if result.faithfulness is not None else "—",
                f"{result.hallucination_rate:.3f}" if result.hallucination_rate is not None else "—",
                str(result.num_samples),
            )
        console.print(table)
