"""
Argumentative-Path RAG 動作確認デモ。

研究計画書 §1 の Q3売上例を使い、以下を実際に動かす:
  1. 文書をインデックス（LLMがノード・エッジを抽出）
  2. WHY型クエリで検索・回答生成

使い方:
    cd argumentative-path-rag
    cp .env.example .env          # OPENAI_API_KEY を設定
    uv run python scripts/demo.py

グラフDB: --store neo4j でNeo4j、デフォルトはインメモリ(NetworkX)
    uv run python scripts/demo.py --store neo4j
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# src/ を Python パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import print as rprint

console = Console()

# ── サンプル文書 ──────────────────────────────────────────────────────────────

SAMPLE_DOC_ID = "q3_sales_report"

SAMPLE_TEXT = """\
当社のQ3売上は前年比12%減少した。
この減少の主要因は電子部品の在庫調整である。
具体的には、半導体不足により製品の生産ラインが停止を余儀なくされた。
なお、為替レートは本四半期を通じて安定していたため、為替影響は軽微であった。
一方、競合X社は同期間に売上を8%伸ばしており、当社との差は拡大している。
この半導体不足は業界全体の構造的問題であり、来四半期も継続する見通しである。
"""

SAMPLE_QUERIES = [
    ("WHY",       "なぜQ3の売上が落ちたのか？"),
    ("EVIDENCE",  "売上減少を支持する根拠は何か？"),
    ("ASSUMPTION","この分析はどのような前提に基づいているか？"),
    ("WHAT",      "Q3売上の変化率はいくらか？"),
]


# ── メイン ────────────────────────────────────────────────────────────────────

def main(store_type: str = "memory") -> None:
    console.print(Panel.fit(
        "[bold cyan]Argumentative-Path RAG — 動作確認デモ[/]",
        border_style="cyan",
    ))

    # ── 設定 & クライアント ───────────────────────────────────────────────────
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]❌ OPENAI_API_KEY が設定されていません。.env を確認してください。[/]")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    # ── グラフストア選択 ───────────────────────────────────────────────────────
    from ap_rag.graph.networkx_store import NetworkXGraphStore
    from ap_rag.graph.neo4j_store import Neo4jGraphStore

    if store_type == "neo4j":
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        neo4j_pass = os.environ.get("NEO4J_PASSWORD", "ap_rag_dev")
        console.print(f"[green]📦 グラフストア: Neo4j ({neo4j_uri})[/]")
        store = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_pass)
    else:
        console.print("[green]📦 グラフストア: NetworkX（インメモリ）[/]")
        store = NetworkXGraphStore()

    # ── インデックスパイプライン ───────────────────────────────────────────────
    from ap_rag.indexing.chunker import SentenceChunker
    from ap_rag.indexing.classifier import NodeClassifier
    from ap_rag.indexing.extractor import EdgeExtractor
    from ap_rag.indexing.pipeline import IndexingPipeline

    console.print(Rule("[bold]Step 1: 文書インデックス[/]"))
    console.print(f"[dim]文書ID: {SAMPLE_DOC_ID}[/]")
    console.print(Panel(SAMPLE_TEXT.strip(), title="サンプル文書", border_style="dim"))

    indexing_pipeline = IndexingPipeline(
        chunker=SentenceChunker(max_tokens=300),
        classifier=NodeClassifier(client=client, model="gpt-4o"),
        extractor=EdgeExtractor(client=client, model="gpt-4o"),
        store=store,
        show_progress=True,
    )

    with console.status("[cyan]LLMでノード・エッジを抽出中...[/]"):
        result = indexing_pipeline.run(SAMPLE_TEXT, doc_id=SAMPLE_DOC_ID)

    console.print(f"[green]✅ インデックス完了[/]")
    _print_graph_stats(result)

    # ── 構築されたグラフを表示 ────────────────────────────────────────────────
    console.print(Rule("[bold]Step 2: 抽出されたグラフ[/]"))
    _print_graph(result.graph)

    # ── クエリ実行 ────────────────────────────────────────────────────────────
    from ap_rag.retrieval.query_classifier import QueryClassifier
    from ap_rag.retrieval.traversal import GraphTraverser
    from ap_rag.retrieval.context_builder import ContextBuilder
    from ap_rag.generation.generator import AnswerGenerator
    from ap_rag.pipeline import ArgumentativeRAGPipeline

    rag_pipeline = ArgumentativeRAGPipeline(
        store=store,
        query_classifier=QueryClassifier(client=client, model="gpt-4o-mini"),
        traverser=GraphTraverser(store=store),
        context_builder=ContextBuilder(max_nodes=15),
        generator=AnswerGenerator(client=client, model="gpt-4o-mini"),
    )

    console.print(Rule("[bold]Step 3: クエリ実行[/]"))

    for query_type_label, query in SAMPLE_QUERIES:
        console.print(f"\n[bold yellow]❓ [{query_type_label}] {query}[/]")
        with console.status("[cyan]検索・回答生成中...[/]"):
            gen_result = rag_pipeline.query(query, doc_id=SAMPLE_DOC_ID)

        console.print(f"[bold green]📌 クエリ型判定:[/] {gen_result.retrieval_context.query_type.value}")
        console.print(f"[bold green]📚 取得ノード数:[/] {gen_result.retrieval_context.num_nodes}")
        console.print(Panel(
            gen_result.answer,
            title=f"[bold]回答[/]",
            border_style="green",
        ))

    if store_type == "neo4j":
        store.close()

    console.print(Rule())
    console.print("[bold cyan]✅ デモ完了！[/]")


def _print_graph_stats(result) -> None:
    stats = result.graph.stats()
    table = Table(title="グラフ統計", show_header=True, header_style="bold magenta")
    table.add_column("項目")
    table.add_column("値", justify="right")
    table.add_row("チャンク数", str(result.num_chunks))
    table.add_row("ノード数（合計）", str(stats["total_nodes"]))
    table.add_row("エッジ数（合計）", str(stats["total_edges"]))
    for node_type, count in stats["nodes_by_type"].items():
        if count > 0:
            table.add_row(f"  → {node_type}", str(count))
    console.print(table)


def _print_graph(graph) -> None:
    from ap_rag.models.taxonomy import NodeType

    NODE_COLORS = {
        NodeType.CLAIM:      "[bold cyan]CLAIM[/]",
        NodeType.EVIDENCE:   "[bold green]EVIDENCE[/]",
        NodeType.ASSUMPTION: "[bold yellow]ASSUMPTION[/]",
        NodeType.CONCLUSION: "[bold magenta]CONCLUSION[/]",
        NodeType.CAVEAT:     "[dim]CAVEAT[/]",
        NodeType.CONTRAST:   "[bold red]CONTRAST[/]",
        NodeType.DEFINITION: "[blue]DEFINITION[/]",
    }

    table = Table(title="抽出ノード一覧", show_header=True, header_style="bold")
    table.add_column("型", width=12)
    table.add_column("テキスト")
    table.add_column("chunk", justify="right", width=6)

    for node in sorted(graph.nodes.values(), key=lambda n: (n.source_chunk_idx, n.node_type.value)):
        table.add_row(
            NODE_COLORS.get(node.node_type, node.node_type.value),
            node.text[:60] + ("..." if len(node.text) > 60 else ""),
            str(node.source_chunk_idx),
        )
    console.print(table)

    if graph.edges:
        edge_table = Table(title="抽出エッジ一覧", show_header=True, header_style="bold")
        edge_table.add_column("エッジ型", width=14)
        edge_table.add_column("from (テキスト冒頭)")
        edge_table.add_column("→ to (テキスト冒頭)")
        edge_table.add_column("confidence", justify="right")

        for edge in graph.edges.values():
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if src and tgt:
                edge_table.add_row(
                    edge.edge_type.value,
                    src.text[:30] + "...",
                    tgt.text[:30] + "...",
                    f"{edge.confidence:.2f}",
                )
        console.print(edge_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argumentative-Path RAG デモ")
    parser.add_argument(
        "--store",
        choices=["memory", "neo4j"],
        default="memory",
        help="グラフストアの選択（デフォルト: memory）",
    )
    parser.add_argument("--debug", action="store_true", help="デバッグログを表示")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    main(store_type=args.store)
