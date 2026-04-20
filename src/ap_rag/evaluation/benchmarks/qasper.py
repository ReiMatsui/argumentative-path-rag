"""
QASPER ベンチマーク実行器。

研究計画書 §4.3「ベンチマーク: QASPER（必須）— 科学論文長文書QA」。

QASPER データセット:
  - HuggingFace: allenai/qasper
  - 長文科学論文に対するQAで、WHY/WHAT/HOW/EVIDENCE タイプが豊富

使い方:
    runner = QASPERRunner(pipeline, indexing_pipeline, client)
    result = runner.run(num_papers=10, num_questions=50)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ap_rag.evaluation.metrics import EvaluationSample

logger = logging.getLogger(__name__)

# クエリ型推定のための簡易ルール
_QUERY_TYPE_RULES = [
    (["why", "なぜ", "reason", "cause", "理由", "原因"], "WHY"),
    (["how", "どうやって", "どのように", "手順", "方法", "procedure"], "HOW"),
    (["evidence", "根拠", "証拠", "support", "prove"], "EVIDENCE"),
    (["assumption", "前提", "premise", "assume"], "ASSUMPTION"),
]


def infer_query_type(question: str) -> str:
    """質問文からクエリ型を簡易推定する（LLMを使わない高速版）。"""
    q_lower = question.lower()
    for keywords, qt in _QUERY_TYPE_RULES:
        if any(kw in q_lower for kw in keywords):
            return qt
    return "WHAT"


@dataclass
class QASPERSample:
    """QASPER の1サンプル。

    Attributes:
        paper_id: 論文ID。
        question: 質問文。
        answer: 正解回答文字列。
        full_text: 論文全文。
        evidence: gold evidence スパンのリスト。QASPER 公式の Evidence-F1
            （取得された evidence と gold evidence の token-F1）の計算に使う。
            Unanswerable 質問や空アノテーションの場合は空リスト。
    """
    paper_id: str
    question: str
    answer: str
    full_text: str
    evidence: list[str] = field(default_factory=list)


class QASPERLoader:
    """QASPER データセットをロードする。

    Args:
        split: "train" / "validation" / "test"
        max_papers: ロードする論文数の上限（None = 全件）。
    """

    def __init__(self, split: str = "validation", max_papers: int | None = None) -> None:
        self._split = split
        self._max_papers = max_papers

    def load(self) -> list[QASPERSample]:
        """HuggingFace から QASPER をロードしてサンプルを返す。"""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "datasets パッケージが必要です: pip install datasets"
            ) from e

        logger.info("QASPER データセットをロード中 (split=%s)...", self._split)
        try:
            # datasets 2.x: trust_remote_code が必要
            dataset = load_dataset(
                "allenai/qasper", split=self._split, trust_remote_code=True
            )
        except TypeError:
            # datasets 3.x 以降: trust_remote_code 引数が廃止された場合
            dataset = load_dataset("allenai/qasper", split=self._split)

        samples: list[QASPERSample] = []
        paper_count = 0

        for paper in dataset:
            if self._max_papers and paper_count >= self._max_papers:
                break

            paper_id = paper["id"]
            full_text = self._extract_full_text(paper)

            # datasets 2.x は qas が列指向 dict {"question": [...], "answers": [...]}
            # datasets 1.x は qas がレコードリスト [{"question": ..., "answers": ...}, ...]
            qas_raw = paper.get("qas", [])
            qa_records = self._normalize_qas(qas_raw)

            for qa in qa_records:
                question = qa.get("question", "").strip()
                if not question:
                    continue

                # 複数アノテーターの回答から最初の有効な回答を使用
                answer = self._extract_answer(qa)
                if not answer:
                    continue

                # gold evidence スパンをアノテーター横断で集める
                evidence = self._extract_evidence(qa)

                samples.append(QASPERSample(
                    paper_id=paper_id,
                    question=question,
                    answer=answer,
                    full_text=full_text,
                    evidence=evidence,
                ))

            paper_count += 1

        logger.info("QASPER ロード完了: %d 論文, %d サンプル", paper_count, len(samples))
        return samples

    @staticmethod
    def _normalize_qas(qas_raw) -> list[dict]:
        """datasets バージョンによる qas 形式の差異を吸収する。

        - 列指向 dict: {"question": [...], "answers": [...]} → レコードリストに変換
        - レコードリスト: [{"question": ..., "answers": ...}] → そのまま返す
        """
        if isinstance(qas_raw, dict):
            # 列指向 dict（datasets 2.x）
            questions = qas_raw.get("question", [])
            answers_list = qas_raw.get("answers", [])
            return [
                {"question": q, "answers": a}
                for q, a in zip(questions, answers_list)
            ]
        # レコードリスト（datasets 1.x）またはその他
        return list(qas_raw)

    @staticmethod
    def _extract_full_text(paper: dict) -> str:
        """論文の全テキストを結合して返す。"""
        parts: list[str] = []

        # タイトル
        title = paper.get("title", "")
        if title:
            parts.append(f"Title: {title}")

        # Abstract
        abstract = paper.get("abstract", "")
        if abstract:
            parts.append(f"Abstract: {abstract}")

        # 本文セクション
        full_text = paper.get("full_text", {})
        for section in full_text.get("section_name", []):
            parts.append(f"\nSection: {section}")
        for paragraphs in full_text.get("paragraphs", []):
            for para in paragraphs:
                if para:
                    parts.append(para)

        return "\n".join(parts)

    @staticmethod
    def _extract_answer(qa: dict) -> str:
        """QAエントリから回答文字列を抽出する。

        answers は以下の2形式がある:
          - レコードリスト: [{"answer": {"free_form_answer": ..., ...}, ...}]
          - 列指向 dict:   {"answer": [{"free_form_answer": ..., ...}], ...}
        """
        raw_answers = qa.get("answers", [])

        # 列指向 dict の場合 → レコードリストに変換
        if isinstance(raw_answers, dict):
            answer_dicts = raw_answers.get("answer", [])
            answer_records = [{"answer": a} for a in answer_dicts]
        else:
            answer_records = list(raw_answers)

        for answer_data in answer_records:
            answer = answer_data.get("answer", {})
            if not isinstance(answer, dict):
                continue

            # 1) Free-form answer（最優先）
            free_form = (answer.get("free_form_answer") or "").strip()
            if free_form and free_form.lower() not in ("yes", "no", "unanswerable"):
                return free_form

            # 2) Extractive spans（本文抜き出し型）— QASPER で最も多い形式
            spans = answer.get("extractive_spans") or []
            if isinstance(spans, (list, tuple)) and spans:
                # 複数スパンを連結して1つの回答文字列にする
                joined = " ... ".join(s.strip() for s in spans if isinstance(s, str) and s.strip())
                if joined:
                    return joined

            # 3) Yes/No answer
            yes_no = (answer.get("yes_no_answer") or "").strip()
            if yes_no.lower() in ("yes", "no"):
                return yes_no

        return ""

    @staticmethod
    def _extract_evidence(qa: dict) -> list[str]:
        """QAエントリから gold evidence スパンのリストを抽出する。

        QASPER のアノテーションは複数アノテーターを持つ。各アノテーターは
        `evidence` フィールドに本文抜粋のリストを持つ。ここでは全アノテーターの
        evidence を結合し、重複除去したリストを返す。

        `answers` は以下の2形式がある（`_extract_answer` と同じ）:
          - レコードリスト: [{"answer": {...}, ...}]
          - 列指向 dict:   {"answer": [{"evidence": ...}], ...}

        `evidence` 自体も次の形のどちらかになりうる:
          - list[str]     : スパン文字列のリスト
          - dict          : {"evidence": [...]} のようなラッパ（稀）
        """
        raw_answers = qa.get("answers", [])

        if isinstance(raw_answers, dict):
            answer_dicts = raw_answers.get("answer", [])
            answer_records = [{"answer": a} for a in answer_dicts]
        else:
            answer_records = list(raw_answers)

        collected: list[str] = []
        seen: set[str] = set()

        for answer_data in answer_records:
            answer = answer_data.get("answer", {})
            if not isinstance(answer, dict):
                continue

            raw_evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
            if isinstance(raw_evidence, dict):
                # まれにネストした dict の場合
                raw_evidence = raw_evidence.get("evidence", [])

            if not isinstance(raw_evidence, (list, tuple)):
                continue

            for span in raw_evidence:
                if not isinstance(span, str):
                    continue
                span = span.strip()
                if not span:
                    continue
                # "FLOAT SELECTED: ..." のような表マーカーはスキップ
                # （QASPER 公式 evaluator は除外しないが、text-F1 では雑音）
                if span in seen:
                    continue
                seen.add(span)
                collected.append(span)

        return collected


class QASPERRunner:
    """QASPER でArgumentative-Path RAGを評価する実行器。

    Args:
        rag_pipeline: ArgumentativeRAGPipeline インスタンス。
        indexing_pipeline: IndexingPipeline インスタンス。
        num_papers: 使用する論文数。
        num_questions: 論文あたりの最大質問数。
        split: QASPER の分割（"validation" 推奨）。
        query_classifier: クエリ型分類器。指定されれば LLM ベースの分類を
            使い、パイプライン内の再分類と ``per_query_type`` 集計の型が
            一致するようにする。None の場合はキーワード規則にフォールバック。
    """

    def __init__(
        self,
        rag_pipeline: Any,
        indexing_pipeline: Any,
        num_papers: int = 10,
        num_questions_per_paper: int = 5,
        split: str = "validation",
        query_classifier: Any | None = None,
    ) -> None:
        self._rag = rag_pipeline
        self._indexer = indexing_pipeline
        self._num_papers = num_papers
        self._num_questions = num_questions_per_paper
        self._split = split
        self._query_classifier = query_classifier

    def load_samples(self) -> list[EvaluationSample]:
        """QASPER から EvaluationSample のリストを生成する。

        各論文をインデックスし、QAペアを EvaluationSample に変換する。
        クエリ型は `self._query_classifier` があれば LLM 分類を使い、
        なければキーワード規則（``infer_query_type``）にフォールバックする。
        """
        loader = QASPERLoader(split=self._split, max_papers=self._num_papers)
        raw_samples = loader.load()

        # 論文ごとにインデックス（重複は skip）
        indexed_papers: set[str] = set()
        eval_samples: list[EvaluationSample] = []
        paper_question_count: dict[str, int] = {}

        # 同じ質問を複数回分類しないよう簡易キャッシュ
        qtype_cache: dict[str, str] = {}

        def _classify(question: str) -> str:
            if question in qtype_cache:
                return qtype_cache[question]
            if self._query_classifier is None:
                qt = infer_query_type(question)
            else:
                try:
                    qt = self._query_classifier.classify(question).value
                except Exception as e:
                    logger.warning(
                        "QueryClassifier 失敗 (規則ベースに fallback): %s", e
                    )
                    qt = infer_query_type(question)
            qtype_cache[question] = qt
            return qt

        for raw in raw_samples:
            # 質問数の上限チェック
            count = paper_question_count.get(raw.paper_id, 0)
            if count >= self._num_questions:
                continue

            # 論文をインデックス（初回のみ）
            if raw.paper_id not in indexed_papers:
                logger.info("論文をインデックス中: %s", raw.paper_id)
                try:
                    self._indexer.run(raw.full_text, doc_id=raw.paper_id)
                    indexed_papers.add(raw.paper_id)
                except Exception as e:
                    logger.warning("インデックス失敗: %s — %s", raw.paper_id, e)
                    continue

            eval_samples.append(EvaluationSample(
                question=raw.question,
                ground_truth=raw.answer,
                predicted_answer="",  # 評価時に埋める
                retrieved_contexts=[],
                doc_id=raw.paper_id,
                query_type=_classify(raw.question),
            ))
            paper_question_count[raw.paper_id] = count + 1

        logger.info("QASPER サンプル準備完了: %d 件", len(eval_samples))
        return eval_samples
