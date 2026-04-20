"""
クイック評価スクリプト — ArgumentativeRAG vs BM25 ベースライン。

合成QAデータ（Q3売上レポート）を使い、2システムを比較する:
  - ArgumentativeRAG  : グラフ構造 + クエリ型適応BFS
  - BM25RAG           : キーワードマッチベース

計測指標:
  - EM (Exact Match)
  - F1 (トークン重複)
  - Faithfulness (LLM-as-judge)
  - Hallucination Rate (LLM-as-judge)

使い方:
    cd argumentative-path-rag
    uv run python scripts/quick_eval.py

オプション:
    --no-judge    LLM-as-judge をスキップ（高速・低コスト）
    --debug       デバッグログを表示
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

# ── サンプル文書 ──────────────────────────────────────────────────────────────

DOC_ID = "q3_sales_report"

DOCUMENT_TEXT = """\
当社のQ3売上は前年比12%減少した。
この減少の主要因は電子部品の在庫調整である。
具体的には、半導体不足により製品の生産ラインが停止を余儀なくされた。
なお、為替レートは本四半期を通じて安定していたため、為替影響は軽微であった。
一方、競合X社は同期間に売上を8%伸ばしており、当社との差は拡大している。
この半導体不足は業界全体の構造的問題であり、来四半期も継続する見通しである。
また、コスト削減のため製造拠点の見直しを検討中であり、早ければ来四半期末までに決定する。
"""

# ── 合成QAデータセット ────────────────────────────────────────────────────────

# (question, ground_truth, query_type)
QA_DATASET = [
    # WHY — 因果関係を問う
    ("なぜQ3の売上が減少したのか？",
     "電子部品の在庫調整、特に半導体不足による生産ライン停止が主因である。",
     "WHY"),
    ("なぜ生産ラインが停止したのか？",
     "半導体不足により製品の生産ラインが停止を余儀なくされた。",
     "WHY"),
    ("なぜ競合X社との差が拡大しているのか？",
     "競合X社は売上を8%伸ばしたが、当社は12%減少したため差が拡大している。",
     "WHY"),

    # WHAT — 事実・数値を問う
    ("Q3売上の前年比変化率はいくらか？",
     "前年比12%減少した。",
     "WHAT"),
    ("競合X社のQ3売上変化率はいくらか？",
     "競合X社は同期間に売上を8%伸ばした。",
     "WHAT"),
    ("為替レートの影響はどうだったか？",
     "為替レートは安定していたため為替影響は軽微であった。",
     "WHAT"),

    # EVIDENCE — 根拠・証拠を問う
    ("売上減少の根拠として示されている事実は何か？",
     "半導体不足により製品の生産ラインが停止した事実が根拠として示されている。",
     "EVIDENCE"),
    ("在庫調整が主因であることを支持する証拠は何か？",
     "半導体不足で生産ラインが停止を余儀なくされたことが具体的証拠である。",
     "EVIDENCE"),

    # ASSUMPTION — 前提を問う
    ("この分析はどのような前提に基づいているか？",
     "為替レートが安定していたという前提のもとで分析が行われている。",
     "ASSUMPTION"),
    ("半導体不足が構造的問題という判断の前提は何か？",
     "業界全体の構造的問題であり来四半期も継続するという見通しが前提となっている。",
     "ASSUMPTION"),

    # HOW — 手段・方法を問う
    ("コスト削減をどのように進める予定か？",
     "製造拠点の見直しを検討中であり、早ければ来四半期末までに決定する予定である。",
     "HOW"),
    ("売上回復のためにどのような対策が示されているか？",
     "製造拠点の見直しを検討中であり来四半期末までに決定する予定だと示されている。",
     "HOW"),
]


# ── ArgumentativeRAG の構築 ───────────────────────────────────────────────────

def build_argumentative_rag(client):
    """ArgumentativeRAGPipeline を初期化してサンプル文書をインデックスする。"""
    from ap_rag.graph.networkx_store import NetworkXGraphStore
    from ap_rag.indexing.chunker import SentenceChunker
    from ap_rag.indexing.classifier import NodeClassifier
    from ap_rag.indexing.extractor import EdgeExtractor
    from ap_rag.indexing.pipeline import IndexingPipeline
    from ap_rag.retrieval.query_classifier import QueryClassifier
    from ap_rag.retrieval.traversal import GraphTraverser
    from ap_rag.retrieval.context_builder import ContextBuilder
    from ap_rag.generation.generator import AnswerGenerator
    from ap_rag.pipeline import ArgumentativeRAGPipeline

    store = NetworkXGraphStore()
    indexer = IndexingPipeline(
        chunker=SentenceChunker(),
        classifier=NodeClassifier(client=client, model="gpt-4o-mini"),
        extractor=EdgeExtractor(client=client, model="gpt-4o-mini"),
        store=store,
        show_progress=False,
    )

    console.print("[cyan]  → ArgumentativeRAG: 文書をインデックス中...[/]")
    with console.status("[dim]LLMがグラフを構築中...[/]"):
        result = indexer.run(DOCUMENT_TEXT, doc_id=DOC_ID)

    stats = result.graph.stats()
    console.print(
        f"  [green]✓ グラフ構築完了[/] — "
        f"ノード数: [bold]{stats['total_nodes']}[/], "
        f"エッジ数: [bold]{stats['total_edges']}[/]"
    )

    pipeline = ArgumentativeRAGPipeline(
        store=store,
        query_classifier=QueryClassifier(client=client, model="gpt-4o-mini"),
        traverser=GraphTraverser(store=store),
        context_builder=ContextBuilder(max_nodes=15),
        generator=AnswerGenerator(client=client, model="gpt-4o-mini"),
    )
    return pipeline


def build_bm25_rag(client):
    """BM25RAG を初期化してサンプル文書をインデックスする。"""
    from ap_rag.evaluation.baselines import BM25RAG

    rag = BM25RAG(client=client, top_k=3)
    # 文書を文単位で分割してインデックス
    chunks = [line.strip() for line in DOCUMENT_TEXT.strip().splitlines() if line.strip()]
    rag.index(chunks, doc_id=DOC_ID)
    console.print(f"  [green]✓ BM25 インデックス完了[/] — チャンク数: [bold]{len(chunks)}[/]")
    return rag


# ── EvaluationSample の生成 ───────────────────────────────────────────────────

def make_eval_samples():
    from ap_rag.evaluation.metrics import EvaluationSample

    return [
        EvaluationSample(
            question=q,
            ground_truth=gt,
            predicted_answer="",
            retrieved_contexts=[],
            doc_id=DOC_ID,
            query_type=qt,
        )
        for q, gt, qt in QA_DATASET
    ]


# ── 評価実行 ──────────────────────────────────────────────────────────────────

def run_evaluation(system, samples, judge, use_judge: bool):
    from ap_rag.evaluation.evaluator import Evaluator

    evaluator = Evaluator(system=system, judge=judge, show_progress=True)
    return evaluator.evaluate(samples, use_judge=use_judge)


# ── 比較テーブルの表示 ────────────────────────────────────────────────────────

def print_comparison(results: dict, use_judge: bool) -> None:
    console.print(Rule("[bold]評価結果[/]"))

    table = Table(
        title="ArgumentativeRAG vs BM25 比較",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
    )
    table.add_column("システム", style="bold", width=22)
    table.add_column("EM", justify="right", width=8)
    table.add_column("F1", justify="right", width=8)
    if use_judge:
        table.add_column("Faithfulness↑", justify="right", width=16)
        table.add_column("Hallucination↓", justify="right", width=16)
    table.add_column("N", justify="right", width=5)

    for name, result in results.items():
        row = [
            name,
            f"{result.em:.3f}",
            f"{result.f1:.3f}",
        ]
        if use_judge:
            row.append(
                f"{result.faithfulness:.3f}" if result.faithfulness is not None else "—"
            )
            row.append(
                f"{result.hallucination_rate:.3f}" if result.hallucination_rate is not None else "—"
            )
        row.append(str(result.num_samples))
        table.add_row(*row)

    console.print(table)

    # クエリ型別の内訳
    console.print(Rule("[bold]クエリ型別 F1 スコア[/]"))
    type_table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
    )
    type_table.add_column("クエリ型", width=12)
    for name in results:
        type_table.add_column(name, justify="right")

    query_types = sorted({
        qt
        for result in results.values()
        for qt in result.per_query_type.keys()
    })

    for qt in query_types:
        row = [qt]
        for result in results.values():
            f1 = result.per_query_type.get(qt, {}).get("f1")
            row.append(f"{f1:.3f}" if f1 is not None else "—")
        type_table.add_row(*row)

    console.print(type_table)


# ── メイン ────────────────────────────────────────────────────────────────────

def main(use_judge: bool = True) -> None:
    console.print(Panel.fit(
        "[bold cyan]Argumentative-Path RAG — クイック評価[/]\n"
        "[dim]合成QAデータ（Q3売上レポート） × ArgumentativeRAG vs BM25[/]",
        border_style="cyan",
    ))

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]❌ OPENAI_API_KEY が設定されていません。.env を確認してください。[/]")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    # ── インデックス構築 ──────────────────────────────────────────────────────
    console.print(Rule("[bold]Step 1: インデックス構築[/]"))
    ap_rag = build_argumentative_rag(client)
    bm25_rag = build_bm25_rag(client)

    # ── EvaluationSample 生成 ─────────────────────────────────────────────────
    samples = make_eval_samples()
    console.print(
        f"\n[bold]評価サンプル数:[/] {len(samples)} 件 "
        f"([dim]{', '.join(qt for _, _, qt in QA_DATASET[:3])}...[/])\n"
    )

    # ── LLM-as-Judge セットアップ ─────────────────────────────────────────────
    judge = None
    if use_judge:
        from ap_rag.evaluation.metrics import LLMJudge
        judge = LLMJudge(client=client)
        console.print("[cyan]LLM-as-Judge:[/] 有効（Faithfulness + Hallucination Rate）\n")
    else:
        console.print("[dim]LLM-as-Judge: スキップ（--no-judge 指定）[/]\n")

    # ── 評価実行 ──────────────────────────────────────────────────────────────
    results = {}

    console.print(Rule("[bold]Step 2: ArgumentativeRAG 評価[/]"))
    results["ArgumentativeRAG"] = run_evaluation(ap_rag, samples, judge, use_judge)

    console.print(Rule("[bold]Step 3: BM25RAG 評価[/]"))
    results["BM25RAG"] = run_evaluation(bm25_rag, samples, judge, use_judge)

    # ── 結果表示 ──────────────────────────────────────────────────────────────
    print_comparison(results, use_judge)

    # ── サマリー ──────────────────────────────────────────────────────────────
    ap = results["ArgumentativeRAG"]
    bm = results["BM25RAG"]
    f1_delta = ap.f1 - bm.f1

    console.print(Rule())
    if f1_delta > 0:
        console.print(
            f"[bold green]✅ ArgumentativeRAG は BM25 より F1 が "
            f"+{f1_delta:.3f} 高い[/]"
        )
    else:
        console.print(
            f"[bold yellow]⚠️  ArgumentativeRAG と BM25 の F1 差: {f1_delta:+.3f}[/]"
        )

    if use_judge and ap.faithfulness is not None and bm.faithfulness is not None:
        faith_delta = ap.faithfulness - bm.faithfulness
        hall_delta = (ap.hallucination_rate or 0.0) - (bm.hallucination_rate or 0.0)
        console.print(
            f"[bold green]📊 Faithfulness 差: {faith_delta:+.3f} / "
            f"Hallucination Rate 差: {hall_delta:+.3f}[/]"
        )

    console.print("[bold cyan]\n✅ 評価完了！[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArgumentativeRAG vs BM25 クイック評価")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="LLM-as-judge をスキップ（高速・低コスト）",
    )
    parser.add_argument("--debug", action="store_true", help="デバッグログを表示")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    main(use_judge=not args.no_judge)
