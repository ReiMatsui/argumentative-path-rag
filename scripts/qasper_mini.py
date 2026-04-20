"""
QASPER ミニ評価スクリプト — ArgumentativeRAG vs BM25 vs Dense。

allenai/qasper データセット（HuggingFace）を使い、
科学論文長文QAで3システム（AP-RAG / BM25 / Dense）を比較する。

デフォルト設定: 論文3件 × 各5問 = 最大15問

使い方:
    cd argumentative-path-rag
    uv run python scripts/qasper_mini.py

オプション:
    --papers N           使用する論文数（デフォルト: 3）
    --questions N        論文あたりの最大質問数（デフォルト: 5）
    --no-judge           LLM-as-judge をスキップ（高速・低コスト）
    --no-dense           DenseRAG をスキップ（埋め込みモデルが重いとき用）
    --dense-model NAME   DenseRAG の埋め込みモデル名
                         （デフォルト: intfloat/e5-mistral-7b-instruct）
    --consistency-runs N 各サンプルを N 回クエリして一貫性スコアを計測
                         （デフォルト: 1 = 計測しない。N 倍の API コストに注意）
    --cross-chunk        チャンク間エッジ抽出を有効化（精度↑・コスト↑）
    --debug              デバッグログを表示
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# ── QASPER ローダ（直接利用） ─────────────────────────────────────────────────

def load_qasper_samples(num_papers: int, num_questions: int, client=None):
    """HuggingFace から QASPER をロードしてサンプルを返す。

    クエリ型は本番と同じ ``QueryClassifier`` （LLM）で一度だけ分類し、
    `EvaluationSample.query_type` に格納する。これによりパイプライン内で
    再分類される型と ``per_query_type`` 集計がズレなくなる。

    Args:
        num_papers: ロードする論文数。
        num_questions: 論文あたりの最大質問数。
        client: openai.OpenAI インスタンス。None の場合はキーワード規則
            ベースのフォールバックを使う（デバッグ用途のみ）。
    """
    from ap_rag.evaluation.benchmarks.qasper import QASPERLoader, infer_query_type
    from ap_rag.evaluation.metrics import EvaluationSample
    from ap_rag.retrieval.query_classifier import QueryClassifier

    console.print(f"[cyan]QASPER をロード中... (論文{num_papers}件 × 各{num_questions}問)[/]")

    loader = QASPERLoader(split="validation", max_papers=num_papers)
    raw_samples = loader.load()

    # 論文ごとの質問数を制限してサンプル生成
    paper_question_count: dict[str, int] = {}
    queued: list = []
    raw_by_paper: dict[str, str] = {}  # paper_id → full_text（インデックス用）

    for raw in raw_samples:
        count = paper_question_count.get(raw.paper_id, 0)
        if count >= num_questions:
            continue
        raw_by_paper[raw.paper_id] = raw.full_text
        queued.append(raw)
        paper_question_count[raw.paper_id] = count + 1

    # クエリ型分類（LLM）— 同じ質問は 1 回しか分類しないようキャッシュ
    classifier = QueryClassifier(client=client) if client is not None else None
    qtype_cache: dict[str, str] = {}

    def _classify(q: str) -> str:
        if q in qtype_cache:
            return qtype_cache[q]
        if classifier is None:
            qt = infer_query_type(q)
        else:
            try:
                qt = classifier.classify(q).value
            except Exception as e:
                logger.warning("QueryClassifier 失敗 (fallback to 規則): %s", e)
                qt = infer_query_type(q)
        qtype_cache[q] = qt
        return qt

    if classifier is not None:
        console.print(
            f"  [dim]LLM クエリ型分類中... ({len(queued)} 件)[/]"
        )

    eval_samples: list = []
    for raw in queued:
        eval_samples.append(EvaluationSample(
            question=raw.question,
            ground_truth=raw.answer,
            predicted_answer="",
            retrieved_contexts=[],
            doc_id=raw.paper_id,
            query_type=_classify(raw.question),
            gold_evidence=raw.evidence,
        ))

    console.print(
        f"  [green]✓ ロード完了[/] — "
        f"論文 {len(raw_by_paper)} 件, QAペア {len(eval_samples)} 件"
    )

    # クエリ型の分布を表示
    from collections import Counter
    qt_dist = Counter(s.query_type for s in eval_samples)
    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(qt_dist.items()))
    console.print(f"  [dim]クエリ型分布 (LLM): {dist_str}[/]")

    return eval_samples, raw_by_paper


# ── ArgumentativeRAG の構築・インデックス ─────────────────────────────────────

def build_and_index_argumentative_rag(
    client,
    raw_by_paper: dict[str, str],
    embedding_model: str,
    embedding_device: str,
    use_cross_chunk: bool = False,
    shared_encoder=None,
    classifier_model: str = "gpt-4o-mini",
    generator_model: str = "gpt-4o-mini",
):
    """ArgumentativeRAGPipeline を構築し、全論文をインデックスする。

    Args:
        shared_encoder: 指定された場合、EmbeddingNodeSelector と
            CrossChunkEdgeExtractor の両方にこの encoder を注入し、
            sentence-transformers / torch の遅延ロードを完全にスキップする
            （`OpenAIEncoder` など SentenceTransformer 互換アダプタに対応）。
    """
    from ap_rag.graph.networkx_store import NetworkXGraphStore
    from ap_rag.indexing.chunker import SentenceChunker
    from ap_rag.indexing.classifier import NodeClassifier
    from ap_rag.indexing.cross_chunk import CrossChunkEdgeExtractor
    from ap_rag.indexing.extractor import EdgeExtractor
    from ap_rag.indexing.pipeline import IndexingPipeline
    from ap_rag.retrieval.query_classifier import QueryClassifier
    from ap_rag.retrieval.traversal import GraphTraverser
    from ap_rag.retrieval.context_builder import ContextBuilder
    from ap_rag.retrieval.embedding_selector import EmbeddingNodeSelector
    from ap_rag.generation.generator import AnswerGenerator
    from ap_rag.pipeline import ArgumentativeRAGPipeline

    console.print(f"  [dim]入口選定モデル: {embedding_model} (device={embedding_device})[/]")
    node_selector = EmbeddingNodeSelector(
        model_name=embedding_model,
        device=embedding_device,
        top_k=10,
        encoder=shared_encoder,
    )

    edge_extractor = EdgeExtractor(client=client, model=classifier_model)
    cross_chunk = None
    if use_cross_chunk:
        console.print(
            f"  [dim]チャンク間エッジ抽出: 有効 (model={embedding_model})[/]"
        )
        # node_selector と同じ encoder を共有してモデルを二重ロードしない
        encoder_for_cross = shared_encoder or node_selector.get_encoder()
        cross_chunk = CrossChunkEdgeExtractor(
            edge_extractor=edge_extractor,
            encoder=encoder_for_cross,
            embedding_model=embedding_model,
            device=embedding_device,
            top_k=5,
            batch_size=8,
        )

    store = NetworkXGraphStore()
    indexer = IndexingPipeline(
        chunker=SentenceChunker(),
        classifier=NodeClassifier(client=client, model=classifier_model),
        extractor=edge_extractor,
        store=store,
        # チャンク単位の進行状況バーを表示（reasoning モデルは 1 チャンクあたり
        # 数秒〜十数秒かかるため、進捗が見えないと不安になる）。
        show_progress=True,
        cross_chunk_extractor=cross_chunk,
    )

    total_nodes, total_edges = 0, 0
    for i, (paper_id, full_text) in enumerate(raw_by_paper.items(), 1):
        short_id = paper_id[:16] + "..."
        console.print(f"  [cyan]({i}/{len(raw_by_paper)}) インデックス中: {short_id}[/]")
        # IndexingPipeline 側が rich.Progress で進捗バーを出すので、
        # ここで console.status を被せると Live が入れ子になって壊れる。
        result = indexer.run(full_text, doc_id=paper_id)
        stats = result.graph.stats()
        total_nodes += stats["total_nodes"]
        total_edges += stats["total_edges"]
        console.print(
            f"    [green]✓[/] ノード {stats['total_nodes']} / エッジ {stats['total_edges']}"
        )

    console.print(
        f"  [bold green]合計: ノード {total_nodes} / エッジ {total_edges}[/]"
    )

    pipeline = ArgumentativeRAGPipeline(
        store=store,
        query_classifier=QueryClassifier(client=client, model=generator_model),
        traverser=GraphTraverser(store=store),
        context_builder=ContextBuilder(max_nodes=15),
        generator=AnswerGenerator(client=client, model=generator_model),
        node_selector=node_selector,
    )
    return pipeline


# ── BM25RAG の構築・インデックス ──────────────────────────────────────────────

def build_and_index_bm25(client, raw_by_paper: dict[str, str], generator_model: str = "gpt-4o-mini"):
    """BM25RAG を構築し、論文テキストをチャンク分割してインデックスする。"""
    from ap_rag.evaluation.baselines import BM25RAG
    from ap_rag.indexing.chunker import SentenceChunker

    rag = BM25RAG(client=client, top_k=5, model=generator_model)
    chunker = SentenceChunker()
    total_chunks = 0

    for paper_id, full_text in raw_by_paper.items():
        chunks = chunker.chunk(full_text, doc_id=paper_id)
        texts = [c.text for c in chunks]
        rag.index(texts, doc_id=paper_id)
        total_chunks += len(texts)

    console.print(f"  [green]✓ BM25 インデックス完了[/] — 総チャンク数: {total_chunks}")
    return rag


# ── DenseRAG の構築・インデックス ─────────────────────────────────────────────

def build_and_index_dense(
    client,
    raw_by_paper: dict[str, str],
    embedding_model: str,
    embedding_device: str,
    shared_encoder=None,
    generator_model: str = "gpt-4o-mini",
):
    """DenseRAG（埋め込み検索）を構築し、論文テキストをインデックスする。

    ``shared_encoder`` を渡すと sentence-transformers の遅延ロードを
    スキップし、同一 encoder で入口選定側とチャンク埋め込みを共有できる。
    """
    from ap_rag.evaluation.baselines import DenseRAG
    from ap_rag.indexing.chunker import SentenceChunker

    console.print(
        f"  [dim]Dense 埋め込みモデル: {embedding_model} (device={embedding_device})[/]"
    )
    rag = DenseRAG(
        client=client,
        top_k=5,
        embedding_model=embedding_model,
        device=embedding_device,
        encoder=shared_encoder,
        model=generator_model,
    )
    chunker = SentenceChunker()
    total_chunks = 0

    for paper_id, full_text in raw_by_paper.items():
        chunks = chunker.chunk(full_text, doc_id=paper_id)
        texts = [c.text for c in chunks]
        rag.index(texts, doc_id=paper_id)
        total_chunks += len(texts)

    console.print(f"  [green]✓ Dense インデックス完了[/] — 総チャンク数: {total_chunks}")
    return rag


# ── 評価実行 ──────────────────────────────────────────────────────────────────

def run_evaluation(
    system,
    samples,
    judge,
    use_judge: bool,
    label: str,
    consistency_runs: int = 1,
    include_per_sample: bool = False,
):
    from ap_rag.evaluation.evaluator import Evaluator
    console.print(f"[bold]評価中: {label}[/]")
    evaluator = Evaluator(
        system=system,
        judge=judge,
        show_progress=True,
        consistency_runs=consistency_runs,
        include_per_sample=include_per_sample,
    )
    return evaluator.evaluate(samples, use_judge=use_judge)


# ── 比較テーブルの表示 ────────────────────────────────────────────────────────

def print_comparison(results: dict, use_judge: bool) -> None:
    console.print(Rule("[bold]評価結果[/]"))

    # 一貫性列を表示するかは結果のいずれかが answer_consistency を持つかで判断
    show_consistency = any(
        r.answer_consistency is not None for r in results.values()
    )
    show_evidence_f1 = any(
        getattr(r, "evidence_f1", None) is not None for r in results.values()
    )

    # 全体比較
    table = Table(
        title="QASPER ミニ評価: AP-RAG vs BM25 vs Dense",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
    )
    table.add_column("システム", style="bold", width=22)
    table.add_column("EM", justify="right", width=8)
    table.add_column("F1", justify="right", width=8)
    if show_evidence_f1:
        table.add_column("Evidence-F1↑", justify="right", width=14)
    if use_judge:
        table.add_column("Correctness↑", justify="right", width=14)
        table.add_column("Faithfulness↑", justify="right", width=14)
        table.add_column("Hallucination↓", justify="right", width=15)
    if show_consistency:
        table.add_column("Consistency↑", justify="right", width=14)
    table.add_column("N", justify="right", width=5)

    for name, result in results.items():
        row = [name, f"{result.em:.3f}", f"{result.f1:.3f}"]
        if show_evidence_f1:
            ef1 = getattr(result, "evidence_f1", None)
            row.append(f"{ef1:.3f}" if ef1 is not None else "—")
        if use_judge:
            row.append(
                f"{result.answer_correctness:.3f}"
                if result.answer_correctness is not None else "—"
            )
            row.append(
                f"{result.faithfulness:.3f}" if result.faithfulness is not None else "—"
            )
            row.append(
                f"{result.hallucination_rate:.3f}" if result.hallucination_rate is not None else "—"
            )
        if show_consistency:
            row.append(
                f"{result.answer_consistency:.3f}"
                if result.answer_consistency is not None else "—"
            )
        row.append(str(result.num_samples))
        table.add_row(*row)

    console.print(table)

    # クエリ型別 F1
    console.print(Rule("[bold]クエリ型別 F1 スコア[/]"))
    type_table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    type_table.add_column("クエリ型", width=14)
    for name in results:
        type_table.add_column(name, justify="right")
    type_table.add_column("サンプル数", justify="right", width=10)

    query_types = sorted({
        qt
        for result in results.values()
        for qt in result.per_query_type.keys()
    })

    # サンプル数は最初のシステムの per_query_type から取る
    first_result = next(iter(results.values()))
    for qt in query_types:
        row = [qt]
        for result in results.values():
            f1 = result.per_query_type.get(qt, {}).get("f1")
            row.append(f"{f1:.3f}" if f1 is not None else "—")
        n = first_result.per_query_type.get(qt, {}).get("count", "—")
        row.append(str(int(n)) if isinstance(n, (int, float)) else str(n))
        type_table.add_row(*row)

    console.print(type_table)


# ── サマリコメント ────────────────────────────────────────────────────────────

def print_summary(results: dict, use_judge: bool) -> None:
    console.print(Rule())

    ap = results.get("ArgumentativeRAG")
    if not ap:
        console.print("[bold cyan]\n✅ QASPER ミニ評価完了！[/]")
        return

    # 比較対象（BM25 → Dense の順に強いベースライン）
    for bl_name in ("BM25RAG", "DenseRAG"):
        bl = results.get(bl_name)
        if bl is None:
            continue

        f1_delta = ap.f1 - bl.f1
        if f1_delta > 0:
            console.print(
                f"[bold green]✅ ArgumentativeRAG は {bl_name} より F1 が +{f1_delta:.3f} 高い[/]"
            )
        elif f1_delta < 0:
            console.print(
                f"[bold yellow]⚠️  {bl_name} の方が F1 が {-f1_delta:.3f} 高い[/]"
            )
        else:
            console.print(f"[dim]ArgumentativeRAG と {bl_name} の F1 は同スコア[/]")

        if use_judge and ap.faithfulness is not None and bl.faithfulness is not None:
            faith_delta = ap.faithfulness - bl.faithfulness
            hall_delta = (bl.hallucination_rate or 0.0) - (ap.hallucination_rate or 0.0)
            console.print(
                f"  [cyan]vs {bl_name}:[/] Faithfulness 差 {faith_delta:+.3f} / "
                f"Hallucination 削減 {hall_delta:+.3f}"
            )

        # Evidence-F1 差分（研究計画書 v6 §4.3 新規貢献② の中心指標）
        ap_ef = getattr(ap, "evidence_f1", None)
        bl_ef = getattr(bl, "evidence_f1", None)
        if ap_ef is not None and bl_ef is not None:
            ef_delta = ap_ef - bl_ef
            sign = "+" if ef_delta >= 0 else ""
            console.print(
                f"  [magenta]vs {bl_name}:[/] Evidence-F1 差 {sign}{ef_delta:.3f} "
                f"(AP={ap_ef:.3f} / {bl_name}={bl_ef:.3f})"
            )

    # 一貫性の比較
    if ap.answer_consistency is not None:
        line_parts = [f"AP-RAG {ap.answer_consistency:.3f}"]
        for bl_name in ("BM25RAG", "DenseRAG"):
            bl = results.get(bl_name)
            if bl is not None and bl.answer_consistency is not None:
                line_parts.append(f"{bl_name} {bl.answer_consistency:.3f}")
        console.print("[bold cyan]🔁 回答一貫性 (pair-F1): " + " | ".join(line_parts) + "[/]")

    console.print("[bold cyan]\n✅ QASPER ミニ評価完了！[/]")


# ── メイン ────────────────────────────────────────────────────────────────────

def main(
    num_papers: int,
    num_questions: int,
    use_judge: bool,
    embedding_model: str,
    embedding_device: str,
    use_dense: bool,
    dense_model: str,
    consistency_runs: int,
    use_cross_chunk: bool,
    embedding_backend: str = "sentence-transformers",
    openai_embedding_model: str = "text-embedding-3-small",
    save_json: str | None = None,
    classifier_model: str = "gpt-4o-mini",
    generator_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
) -> None:
    consistency_note = (
        f" / Consistency: {consistency_runs}×" if consistency_runs >= 2 else ""
    )
    cross_chunk_note = " / Cross-chunk: ON" if use_cross_chunk else ""
    backend_note = (
        f" / Backend: {openai_embedding_model}"
        if embedding_backend == "openai"
        else ""
    )
    selector_label = (
        openai_embedding_model
        if embedding_backend == "openai"
        else embedding_model.split('/')[-1]
    )
    console.print(Panel.fit(
        "[bold cyan]QASPER ミニ評価 — Argumentative-Path RAG[/]\n"
        f"[dim]論文 {num_papers} 件 × 各 {num_questions} 問 / "
        f"LLM-as-Judge: {'有効' if use_judge else 'スキップ'} / "
        f"Dense: {'有効' if use_dense else 'スキップ'} / "
        f"入口選定: {selector_label}\n"
        f"classifier={classifier_model} / generator={generator_model} / judge={judge_model}"
        f"{backend_note}{cross_chunk_note}{consistency_note}[/]",
        border_style="cyan",
    ))

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]❌ OPENAI_API_KEY が設定されていません。[/]")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    # ── Step 1: データロード ──────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1: QASPERデータロード[/]"))
    samples, raw_by_paper = load_qasper_samples(num_papers, num_questions, client=client)

    if not samples:
        console.print("[red]❌ サンプルが0件です。ネットワーク接続を確認してください。[/]")
        sys.exit(1)

    # ── Step 1.5: 共有 encoder のセットアップ（OpenAI バックエンド時） ───────
    # 同じ encoder を AP-RAG の入口選定・CrossChunk・DenseRAG の3箇所で使い回し、
    # API 呼び出しをキャッシュ性良く扱う。
    shared_encoder = None
    if embedding_backend == "openai":
        from ap_rag.retrieval.openai_encoder import OpenAIEncoder
        shared_encoder = OpenAIEncoder(
            client=client,
            model=openai_embedding_model,
        )
        # モデル名表示用: 研究計画書上の E5-Mistral 等とは別軸なので
        # 下流の CrossChunk ログなどには分かりやすいラベルを入れておく
        embedding_model_label = f"openai:{openai_embedding_model}"
        console.print(
            f"  [dim]共有 encoder: OpenAI {openai_embedding_model}[/]"
        )
    else:
        embedding_model_label = embedding_model

    # ── Step 2: インデックス構築 ──────────────────────────────────────────────
    console.print(Rule("[bold]Step 2: インデックス構築[/]"))
    console.print("[bold]ArgumentativeRAG:[/]")
    ap_rag = build_and_index_argumentative_rag(
        client, raw_by_paper, embedding_model_label, embedding_device,
        use_cross_chunk=use_cross_chunk,
        shared_encoder=shared_encoder,
        classifier_model=classifier_model,
        generator_model=generator_model,
    )

    console.print("[bold]BM25RAG:[/]")
    bm25_rag = build_and_index_bm25(client, raw_by_paper, generator_model=generator_model)

    dense_rag = None
    if use_dense:
        console.print("[bold]DenseRAG:[/]")
        # DenseRAG は OpenAI 共有 encoder をそのまま使う（sentence-transformers
        # の重依存を避けるため）。--embedding-backend が sentence-transformers
        # の場合のみ dense_model を自分でロードする。
        dense_rag = build_and_index_dense(
            client, raw_by_paper,
            embedding_model_label if embedding_backend == "openai" else dense_model,
            embedding_device,
            shared_encoder=shared_encoder,
            generator_model=generator_model,
        )

    # ── Step 3: LLM-as-Judge セットアップ ────────────────────────────────────
    judge = None
    if use_judge:
        from ap_rag.evaluation.metrics import LLMJudge
        judge = LLMJudge(client=client, model=judge_model)
        console.print(f"\n[cyan]LLM-as-Judge:[/] 有効 (model={judge_model})\n")

    # ── Step 4: 評価実行 ──────────────────────────────────────────────────────
    # --save-json を指定されたときだけ per-sample 記録を保持する
    # （数百件スケールでもメモリが気になるレベルではないが、既定は off）
    include_per_sample = bool(save_json)

    console.print(Rule("[bold]Step 3: 評価実行[/]"))
    results: dict = {}
    results["ArgumentativeRAG"] = run_evaluation(
        ap_rag, samples, judge, use_judge, "ArgumentativeRAG",
        consistency_runs=consistency_runs,
        include_per_sample=include_per_sample,
    )
    results["BM25RAG"] = run_evaluation(
        bm25_rag, samples, judge, use_judge, "BM25RAG",
        consistency_runs=consistency_runs,
        include_per_sample=include_per_sample,
    )
    if dense_rag is not None:
        results["DenseRAG"] = run_evaluation(
            dense_rag, samples, judge, use_judge, "DenseRAG",
            consistency_runs=consistency_runs,
            include_per_sample=include_per_sample,
        )

    # ── Step 5: 結果表示 ──────────────────────────────────────────────────────
    print_comparison(results, use_judge)
    print_summary(results, use_judge)

    # ── Step 6: JSON 保存 (任意) ──────────────────────────────────────────────
    if save_json:
        import json
        from dataclasses import asdict
        dumped = {
            name: {
                **asdict(result),
                # EvaluationResult には raw_scores が含まれる場合あり → 除外
            }
            for name, result in results.items()
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "num_papers": num_papers,
                        "num_questions": num_questions,
                        "use_judge": use_judge,
                        "use_dense": use_dense,
                        "use_cross_chunk": use_cross_chunk,
                        "embedding_backend": embedding_backend,
                        "openai_embedding_model": openai_embedding_model,
                        "consistency_runs": consistency_runs,
                        "classifier_model": classifier_model,
                        "generator_model": generator_model,
                        "judge_model": judge_model,
                    },
                    "results": dumped,
                },
                f, ensure_ascii=False, indent=2,
            )
        console.print(f"[bold cyan]📄 結果を保存: {save_json}[/]")


if __name__ == "__main__":
    # .env を argparse 解釈前にロードして、モデル系フラグの既定値として使う
    from dotenv import load_dotenv
    load_dotenv()
    _env_classifier = os.environ.get("OPENAI_CLASSIFIER_MODEL", "gpt-4o-mini")
    _env_generator = os.environ.get("OPENAI_GENERATOR_MODEL", "gpt-4o-mini")
    _env_judge = os.environ.get("OPENAI_JUDGE_MODEL", _env_generator)
    # reasoning_effort は openai_compat が環境変数から読むので、
    # ここでは CLI フラグ経由で `OPENAI_REASONING_EFFORT` を書き換える形にする。
    _env_reasoning = os.environ.get("OPENAI_REASONING_EFFORT", "low")

    parser = argparse.ArgumentParser(description="QASPER ミニ評価")
    parser.add_argument("--papers",    type=int, default=3,  help="論文数（デフォルト: 3）")
    parser.add_argument("--questions", type=int, default=5,  help="論文あたりの質問数（デフォルト: 5）")
    parser.add_argument("--no-judge",        action="store_true",  help="LLM-as-judge をスキップ")
    parser.add_argument("--no-dense",        action="store_true",  help="DenseRAG ベースラインをスキップ")
    parser.add_argument("--debug",           action="store_true",  help="デバッグログを表示")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/e5-mistral-7b-instruct",
        help="入口選定に使う埋め込みモデル（デフォルト: intfloat/e5-mistral-7b-instruct）",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="intfloat/e5-mistral-7b-instruct",
        help="DenseRAG の埋め込みモデル（デフォルト: intfloat/e5-mistral-7b-instruct）",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="埋め込みモデルのデバイス (デフォルト: cpu)",
    )
    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=1,
        help=(
            "各サンプルを N 回クエリして一貫性スコアを計測（デフォルト: 1 = 計測しない）。"
            "N>=2 のとき API コストが N 倍になる点に注意。"
        ),
    )
    parser.add_argument(
        "--cross-chunk",
        action="store_true",
        help="チャンク境界をまたぐエッジ抽出を有効化（精度↑・埋め込み計算と LLM コスト↑）。",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default="sentence-transformers",
        choices=["sentence-transformers", "openai"],
        help=(
            "埋め込みバックエンド。"
            "'sentence-transformers' は --embedding-model / --dense-model の "
            "HuggingFace モデルを device にロード（既定）。"
            "'openai' は OpenAI Embeddings API を共有 encoder として "
            "AP-RAG / DenseRAG / CrossChunk すべてに注入する。M1 MacBook など "
            "torch を避けたい環境で推奨。"
        ),
    )
    parser.add_argument(
        "--openai-embedding-model",
        type=str,
        default="text-embedding-3-small",
        help=(
            "--embedding-backend openai のとき使う OpenAI Embeddings モデル。"
            "既定: text-embedding-3-small (1536 dim)。"
        ),
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="指定したパスに評価結果を JSON で保存する。GO/NO-GO レポート生成用。",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default=_env_classifier,
        help=(
            "グラフ構築 (NodeClassifier + EdgeExtractor) に使う LLM。"
            f"既定: .env の OPENAI_CLASSIFIER_MODEL (現在: {_env_classifier})。"
        ),
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=_env_generator,
        help=(
            "QueryClassifier / AnswerGenerator (AP / BM25 / Dense) に使う LLM。"
            f"既定: .env の OPENAI_GENERATOR_MODEL (現在: {_env_generator})。"
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=_env_judge,
        help=(
            "LLM-as-Judge に使う LLM。"
            f"既定: .env の OPENAI_JUDGE_MODEL (現在: {_env_judge}, "
            "未設定時は generator と同じ)。"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=_env_reasoning,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "gpt-5 系 / o-series の reasoning_effort。推論トークン量に直結し、"
            "低いほど高速・安価。分類・抽出タスクは 'low' で十分なことが多い。"
            f"既定: .env の OPENAI_REASONING_EFFORT (現在: {_env_reasoning})。"
        ),
    )
    args = parser.parse_args()

    # CLI フラグを openai_compat が参照する環境変数に書き戻すことで、
    # classifier / extractor / generator 等の全コンポーネントに一括で効かせる。
    # （各クラスに reasoning_effort パラメータを足すより配管が少ない）
    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    main(
        num_papers=args.papers,
        num_questions=args.questions,
        use_judge=not args.no_judge,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        use_dense=not args.no_dense,
        dense_model=args.dense_model,
        consistency_runs=args.consistency_runs,
        use_cross_chunk=args.cross_chunk,
        embedding_backend=args.embedding_backend,
        openai_embedding_model=args.openai_embedding_model,
        save_json=args.save_json,
        classifier_model=args.classifier_model,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
    )
