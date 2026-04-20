"""
WHY/HOW 合成セットで AP-RAG vs BM25 vs Dense を評価するスクリプト。

前段:
    python scripts/synthesize_why_how.py --out eval_results/why_how_v1.jsonl

本段:
    python scripts/why_how_eval.py \
        --in eval_results/why_how_v1.jsonl \
        --classifier-model gpt-5-mini \
        --generator-model gpt-5-mini \
        --judge-model gpt-5-mini \
        --save-json eval_results/why_how_main_v1.json

合成済み JSONL に含まれる evidence スパンを QASPER 公式形式の gold
evidence として扱い、Evidence-F1 が計算できるように EvaluationSample
に渡す。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# qasper_mini.py のユーティリティを流用する
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from qasper_mini import (  # type: ignore[import-not-found]
    build_and_index_argumentative_rag,
    build_and_index_bm25,
    build_and_index_dense,
    print_comparison,
    print_summary,
    run_evaluation,
)

console = Console()
logger = logging.getLogger(__name__)


def load_synth_samples(path: str, client):
    """合成 JSONL から EvaluationSample を構築する。

    合成時に記録された `query_type` をそのまま使うため、LLM で再分類しない
    (保存された分布を維持する)。
    """
    from ap_rag.evaluation.metrics import EvaluationSample

    samples: list[EvaluationSample] = []
    raw_by_paper: dict[str, str] = {}

    # 元論文本文はここで別ロードする必要がある
    from ap_rag.evaluation.benchmarks.qasper import QASPERLoader

    paper_ids_needed: set[str] = set()
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)
            paper_ids_needed.add(r["paper_id"])

    console.print(f"[cyan]合成 JSONL から {len(records)} 質問 / {len(paper_ids_needed)} 論文をロード[/]")

    # 必要な論文本文をまとめて引く。QASPERLoader は max_papers だけで
    # 絞る挙動なので、全件舐めて paper_id でフィルタする。
    loader = QASPERLoader(split="validation", max_papers=None)
    all_raw = loader.load()
    for raw in all_raw:
        if raw.paper_id in paper_ids_needed and raw.paper_id not in raw_by_paper:
            raw_by_paper[raw.paper_id] = raw.full_text

    missing = paper_ids_needed - set(raw_by_paper.keys())
    if missing:
        console.print(f"[yellow]⚠ 本文が見つからなかった論文 {len(missing)} 件は除外します[/]")
        records = [r for r in records if r["paper_id"] in raw_by_paper]

    for r in records:
        samples.append(EvaluationSample(
            question=r["question"],
            ground_truth=r["answer"],
            predicted_answer="",
            retrieved_contexts=[],
            doc_id=r["paper_id"],
            query_type=r.get("query_type", "WHY"),
            gold_evidence=r.get("evidence", []),
        ))

    from collections import Counter
    dist = Counter(s.query_type for s in samples)
    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
    console.print(f"  [green]✓ {len(samples)} サンプル / 型分布: {dist_str}[/]")

    return samples, raw_by_paper


def main(
    in_path: str,
    use_judge: bool,
    use_dense: bool,
    embedding_backend: str,
    openai_embedding_model: str,
    use_cross_chunk: bool,
    consistency_runs: int,
    classifier_model: str,
    generator_model: str,
    judge_model: str,
    save_json: str | None,
) -> None:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]❌ OPENAI_API_KEY が設定されていません。[/]")
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    console.print(Panel.fit(
        f"[bold magenta]WHY/HOW 検証 — AP-RAG vs Baselines[/]\n"
        f"[dim]入力: {in_path} / "
        f"classifier={classifier_model} / generator={generator_model} / judge={judge_model}[/]",
        border_style="magenta",
    ))

    # ── Step 1: データロード ──
    console.print(Rule("[bold]Step 1: 合成データロード[/]"))
    samples, raw_by_paper = load_synth_samples(in_path, client)

    if not samples:
        console.print("[red]❌ サンプルが0件。合成 JSONL を確認してください。[/]")
        sys.exit(1)

    # ── Step 2: 共有 encoder ──
    shared_encoder = None
    embedding_model_label = "intfloat/e5-mistral-7b-instruct"
    if embedding_backend == "openai":
        from ap_rag.retrieval.openai_encoder import OpenAIEncoder
        shared_encoder = OpenAIEncoder(client=client, model=openai_embedding_model)
        embedding_model_label = f"openai:{openai_embedding_model}"

    # ── Step 3: 各システムをインデックス ──
    console.print(Rule("[bold]Step 2: インデックス構築[/]"))
    console.print("[bold]ArgumentativeRAG:[/]")
    ap_rag = build_and_index_argumentative_rag(
        client, raw_by_paper, embedding_model_label, "cpu",
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
        dense_rag = build_and_index_dense(
            client, raw_by_paper,
            embedding_model_label,
            "cpu",
            shared_encoder=shared_encoder,
            generator_model=generator_model,
        )

    # ── Step 4: judge ──
    judge = None
    if use_judge:
        from ap_rag.evaluation.metrics import LLMJudge
        judge = LLMJudge(client=client, model=judge_model)
        console.print(f"\n[cyan]LLM-as-Judge:[/] 有効 (model={judge_model})\n")

    # ── Step 5: 評価 ──
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

    # ── Step 6: 表示 ──
    print_comparison(results, use_judge)
    print_summary(results, use_judge)

    # ── Step 7: JSON 保存 ──
    if save_json:
        from dataclasses import asdict
        dumped = {name: asdict(r) for name, r in results.items()}
        out_path = Path(save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "source": in_path,
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
    from dotenv import load_dotenv
    load_dotenv()
    _env_classifier = os.environ.get("OPENAI_CLASSIFIER_MODEL", "gpt-5-mini")
    _env_generator = os.environ.get("OPENAI_GENERATOR_MODEL", "gpt-5-mini")
    _env_judge = os.environ.get("OPENAI_JUDGE_MODEL", _env_generator)
    _env_reasoning = os.environ.get("OPENAI_REASONING_EFFORT", "low")

    parser = argparse.ArgumentParser(description="WHY/HOW 合成セット評価")
    parser.add_argument("--in", dest="in_path", type=str,
                        default="eval_results/why_how_v1.jsonl")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--cross-chunk", action="store_true")
    parser.add_argument("--consistency-runs", type=int, default=1)
    parser.add_argument("--embedding-backend", type=str, default="openai",
                        choices=["sentence-transformers", "openai"])
    parser.add_argument("--openai-embedding-model", type=str,
                        default="text-embedding-3-small")
    parser.add_argument("--classifier-model", type=str, default=_env_classifier,
                        help=f"既定: .env の OPENAI_CLASSIFIER_MODEL (現在: {_env_classifier})")
    parser.add_argument("--generator-model", type=str, default=_env_generator,
                        help=f"既定: .env の OPENAI_GENERATOR_MODEL (現在: {_env_generator})")
    parser.add_argument("--judge-model", type=str, default=_env_judge,
                        help=f"既定: .env の OPENAI_JUDGE_MODEL (現在: {_env_judge})")
    parser.add_argument(
        "--reasoning-effort", type=str, default=_env_reasoning,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "gpt-5 系 / o-series の reasoning_effort。低いほど高速・安価。"
            f"既定: .env の OPENAI_REASONING_EFFORT (現在: {_env_reasoning})。"
        ),
    )
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # CLI フラグを openai_compat 用の環境変数に書き戻す
    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)

    main(
        in_path=args.in_path,
        use_judge=not args.no_judge,
        use_dense=not args.no_dense,
        embedding_backend=args.embedding_backend,
        openai_embedding_model=args.openai_embedding_model,
        use_cross_chunk=args.cross_chunk,
        consistency_runs=args.consistency_runs,
        classifier_model=args.classifier_model,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        save_json=args.save_json,
    )
