"""評価フレームワークのユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ap_rag.evaluation.metrics import (
    EvaluationSample,
    LLMJudge,
    aggregate_results,
    compute_em,
    compute_f1,
    normalize_answer,
)
from ap_rag.evaluation.evaluator import Evaluator
from ap_rag.evaluation.baselines import BM25RAG
from ap_rag.evaluation.ablation import NoisyGraphStore
from ap_rag.evaluation.benchmarks.qasper import infer_query_type as qasper_infer
from ap_rag.models import ArgumentNode, NodeType


# ── normalize_answer ──────────────────────────────────────────────────────────

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_punctuation_removed(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_japanese_punctuation_removed(self):
        result = normalize_answer("Q3売上は12%減少した。")
        assert "。" not in result

    def test_extra_spaces_collapsed(self):
        assert normalize_answer("  hello   world  ") == "hello world"


# ── compute_em ────────────────────────────────────────────────────────────────

class TestComputeEM:
    def test_exact_match(self):
        assert compute_em("12% reduction", "12% reduction") == 1.0

    def test_case_insensitive(self):
        assert compute_em("Q3 Sales Fell", "q3 sales fell") == 1.0

    def test_no_match(self):
        assert compute_em("hello", "world") == 0.0

    def test_japanese_match(self):
        assert compute_em("前年比12%減少", "前年比12%減少") == 1.0


# ── compute_f1 ────────────────────────────────────────────────────────────────

class TestComputeF1:
    def test_perfect_match(self):
        assert compute_f1("hello world", "hello world") == 1.0

    def test_partial_match(self):
        score = compute_f1("Q3 sales fell 12%", "Q3 revenue fell 12 percent")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        assert compute_f1("apple", "orange") == 0.0

    def test_empty_strings(self):
        assert compute_f1("", "") == 1.0

    def test_one_empty(self):
        assert compute_f1("hello", "") == 0.0


# ── aggregate_results ─────────────────────────────────────────────────────────

class TestAggregateResults:
    def _make_sample(self, query_type: str = "WHY") -> EvaluationSample:
        return EvaluationSample(
            question="なぜ？",
            ground_truth="理由A",
            predicted_answer="理由A",
            retrieved_contexts=["コンテキスト"],
            doc_id="doc_001",
            query_type=query_type,
        )

    def test_basic_aggregation(self):
        samples = [self._make_sample("WHY"), self._make_sample("WHAT")]
        result = aggregate_results(
            samples=samples,
            em_scores=[1.0, 0.0],
            f1_scores=[1.0, 0.5],
        )
        assert result.em == pytest.approx(0.5)
        assert result.f1 == pytest.approx(0.75)
        assert result.num_samples == 2
        assert result.faithfulness is None
        assert result.hallucination_rate is None

    def test_with_judge_scores(self):
        samples = [self._make_sample()]
        result = aggregate_results(
            samples=samples,
            em_scores=[1.0],
            f1_scores=[1.0],
            faithfulness_scores=[0.9],
            hallucination_flags=[False],
        )
        assert result.faithfulness == pytest.approx(0.9)
        assert result.hallucination_rate == pytest.approx(0.0)

    def test_per_query_type(self):
        samples = [
            self._make_sample("WHY"),
            self._make_sample("WHY"),
            self._make_sample("WHAT"),
        ]
        result = aggregate_results(
            samples=samples,
            em_scores=[1.0, 0.0, 1.0],
            f1_scores=[1.0, 0.0, 1.0],
        )
        assert "WHY" in result.per_query_type
        assert "WHAT" in result.per_query_type
        assert result.per_query_type["WHY"]["em"] == pytest.approx(0.5)
        assert result.per_query_type["WHAT"]["em"] == pytest.approx(1.0)


# ── LLMJudge ──────────────────────────────────────────────────────────────────

class TestLLMJudge:
    def _make_judge(self, response: str) -> LLMJudge:
        mock_choice = MagicMock()
        mock_choice.message.content = response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return LLMJudge(client=mock_client)

    def _make_sample(self) -> EvaluationSample:
        return EvaluationSample(
            question="q",
            ground_truth="a",
            predicted_answer="pred",
            retrieved_contexts=["context"],
            doc_id="doc",
        )

    def test_hallucination_yes(self):
        judge = self._make_judge("YES")
        assert judge.is_hallucination(self._make_sample()) is True

    def test_hallucination_no(self):
        judge = self._make_judge("NO")
        assert judge.is_hallucination(self._make_sample()) is False

    def test_faithfulness_score(self):
        judge = self._make_judge("0.85")
        score = judge.faithfulness_score(self._make_sample())
        assert score == pytest.approx(0.85)

    def test_faithfulness_invalid_response(self):
        judge = self._make_judge("high")
        score = judge.faithfulness_score(self._make_sample())
        assert score == pytest.approx(0.5)  # フォールバック値


# ── BM25RAG ───────────────────────────────────────────────────────────────────

class TestBM25RAG:
    def _make_client(self, answer: str = "テスト回答") -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = answer
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_basic_retrieval(self):
        client = self._make_client("Q3売上は12%減少した。")
        rag = BM25RAG(client=client, top_k=2)
        rag.index(
            texts=[
                "Q3売上は前年比12%減少した。",
                "電子部品の在庫調整が原因。",
                "競合X社は8%増加した。",
            ],
            doc_id="doc_001",
        )
        result = rag.query("Q3の売上変化は？", doc_id="doc_001")
        assert result.answer != ""
        assert len(result.retrieval_context.nodes) == 2

    def test_empty_index_returns_empty(self):
        client = self._make_client()
        rag = BM25RAG(client=client)
        result = rag.query("質問", doc_id="nonexistent")
        assert result.answer == ""


# ── NoisyGraphStore ───────────────────────────────────────────────────────────

class TestNoisyGraphStore:
    def _make_node(self, node_type: NodeType) -> ArgumentNode:
        return ArgumentNode(
            node_type=node_type,
            text="テキスト",
            source_doc_id="doc",
            source_chunk_idx=0,
        )

    def test_zero_error_rate_no_corruption(self):
        mock_store = MagicMock()
        nodes = [self._make_node(NodeType.CLAIM)] * 10
        mock_store.get_nodes_by_type.return_value = nodes
        noisy = NoisyGraphStore(mock_store, error_rate=0.0)
        result = noisy.get_nodes_by_type("doc", NodeType.CLAIM)
        assert all(n.node_type == NodeType.CLAIM for n in result)

    def test_high_error_rate_causes_corruption(self):
        mock_store = MagicMock()
        nodes = [self._make_node(NodeType.CLAIM)] * 100
        mock_store.get_nodes_by_type.return_value = nodes
        noisy = NoisyGraphStore(mock_store, error_rate=1.0, seed=0)
        result = noisy.get_nodes_by_type("doc", NodeType.CLAIM)
        # error_rate=1.0 なら全てのノード型が変わっているはず
        assert all(n.node_type != NodeType.CLAIM for n in result)


# ── infer_query_type ──────────────────────────────────────────────────────────

class TestInferQueryType:
    @pytest.mark.parametrize("question,expected", [
        ("Why did Q3 revenue fall?", "WHY"),
        ("How do I calculate ROI?", "HOW"),
        ("What is the evidence for this claim?", "EVIDENCE"),
        ("What assumption underlies this?", "ASSUMPTION"),
        ("What is the revenue?", "WHAT"),
        ("Q3売上はいくらか？", "WHAT"),
        ("なぜ売上が落ちたのか？", "WHY"),
    ])
    def test_infer(self, question: str, expected: str):
        assert qasper_infer(question) == expected
