"""
WHY/HOW 検証用データセット合成スクリプト。

QASPER validation の論文本文を LLM に読ませ、AP-RAG の強みが効く
"WHY / HOW / EVIDENCE" 型の質問 + gold evidence スパン + 回答 を
生成する。生成モデルと評価モデルは意図的に分ける (e.g., 生成=gpt-5,
評価=gpt-5-mini) ことで同一モデル自己保証バイアスを避ける。

生成される JSONL の 1 行フォーマット:
    {
      "paper_id": str,
      "question": str,
      "query_type": "WHY" | "HOW" | "EVIDENCE",
      "answer": str,
      "evidence": [str, ...],   # 本文から抜き出した根拠スパン (2~4 個)
      "rationale": str,         # 生成時の推論メモ (分析用)
    }

使い方:
    cd argumentative-path-rag
    python scripts/synthesize_why_how.py \
        --papers 6 --questions-per-paper 5 \
        --generator-model gpt-5 \
        --out eval_results/why_how_v1.jsonl

オプション:
    --papers N                    使用する論文数 (デフォルト: 6)
    --questions-per-paper N       論文あたりの目標質問数 (デフォルト: 5)
    --generator-model NAME        合成に使う LLM (推奨: gpt-5)
    --out PATH                    出力 JSONL のパス
    --min-evidence-spans N        1 質問あたりの最小 evidence スパン数 (デフォルト: 2)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ap_rag.openai_compat import reasoning_kwarg, sampling_kwargs
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


SYNTH_SYSTEM_PROMPT = """\
You are a careful question-set author building an evaluation benchmark for
retrieval-augmented generation systems. You will read an excerpt from a
scientific paper and author high-quality WHY / HOW / EVIDENCE questions
grounded strictly in the excerpt.

Strict rules:
1. Each question MUST require reasoning across at least 2 sentences in the
   excerpt (no one-sentence keyword lookups).
2. Each question MUST be answerable from the excerpt alone. Do NOT invent
   information beyond what is written.
3. Evidence spans MUST be verbatim substrings of the excerpt, exactly as
   they appear (same case, same punctuation). Quote 2-4 spans per question.
4. Distribute question types across WHY / HOW / EVIDENCE.
   - WHY: ask for causal / motivational reasoning ("Why does X hold?",
     "Why did the authors choose A over B?").
   - HOW: ask for a process / mechanism ("How is X computed?",
     "How do the authors verify Y?").
   - EVIDENCE: ask what evidence supports a claim ("What evidence
     supports the claim that ...?").
5. The `answer` field is a 1-2 sentence ground-truth answer in plain
   English, suitable for string-level F1 scoring.
6. Avoid questions whose answer is a bare name / number; prefer questions
   whose answer spans a short phrase grounded in the argument structure.

Output JSON only, matching the schema.
"""


SYNTH_USER_TEMPLATE = """\
Paper ID: {paper_id}
Target question count for this excerpt: {n_questions}
Minimum evidence spans per question: {min_spans}

Excerpt:
\"\"\"
{excerpt}
\"\"\"
"""


try:
    from pydantic import BaseModel, Field

    class SynthQuestion(BaseModel):
        question: str
        query_type: str  # WHY / HOW / EVIDENCE
        answer: str
        evidence: list[str] = Field(default_factory=list)
        rationale: str = ""

    class SynthBatch(BaseModel):
        questions: list[SynthQuestion]

    HAS_PYDANTIC = True
except Exception:  # pragma: no cover
    HAS_PYDANTIC = False


def _select_excerpt(full_text: str, max_chars: int = 6000) -> str:
    """長い論文本文から、最も "議論が濃い" 中盤部分を切り出す。

    方針: 先頭 20% は abstract / intro で事実記述が多いのでスキップ、
    続く ~50% のウィンドウ (methods + results + discussion の中心) を返す。
    """
    n = len(full_text)
    if n <= max_chars:
        return full_text
    start = int(n * 0.2)
    end = min(n, start + max_chars)
    return full_text[start:end]


def synthesize_for_paper(
    client: Any,
    paper_id: str,
    full_text: str,
    n_questions: int,
    min_spans: int,
    model: str,
) -> list[dict]:
    """1 本の論文から n_questions 件の WHY/HOW/EVIDENCE 質問を生成する。"""
    if not HAS_PYDANTIC:
        raise RuntimeError("pydantic が必要です (pip install pydantic)")

    excerpt = _select_excerpt(full_text)
    user_msg = SYNTH_USER_TEMPLATE.format(
        paper_id=paper_id,
        n_questions=n_questions,
        min_spans=min_spans,
        excerpt=excerpt,
    )

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=SynthBatch,
        **sampling_kwargs(model, temperature=0.2),
        **reasoning_kwarg(model),
    )
    batch: SynthBatch = response.choices[0].message.parsed  # type: ignore[assignment]

    results: list[dict] = []
    for q in batch.questions:
        # Evidence スパンは本文に部分一致するもののみ採用する
        # (生成器が微妙にリライトしてくるケースをフィルタ)。
        verified_evidence = [e for e in q.evidence if e.strip() and e.strip() in excerpt]
        if len(verified_evidence) < min_spans:
            logger.warning(
                "paper=%s question=%r: 検証済み evidence %d < %d → skip",
                paper_id, q.question[:60], len(verified_evidence), min_spans,
            )
            continue
        results.append(
            {
                "paper_id": paper_id,
                "question": q.question.strip(),
                "query_type": q.query_type.upper().strip(),
                "answer": q.answer.strip(),
                "evidence": verified_evidence,
                "rationale": q.rationale.strip(),
            }
        )

    return results


def main(
    num_papers: int,
    questions_per_paper: int,
    min_spans: int,
    model: str,
    out_path: str,
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
        f"[bold cyan]WHY/HOW 合成 — from QASPER[/]\n"
        f"[dim]論文 {num_papers} × 質問 {questions_per_paper} / "
        f"generator={model} / min_spans={min_spans}[/]",
        border_style="cyan",
    ))

    from ap_rag.evaluation.benchmarks.qasper import QASPERLoader

    # QASPER validation から重複排除して論文本文だけ取り出す
    loader = QASPERLoader(split="validation", max_papers=num_papers)
    raw = loader.load()
    paper_texts: dict[str, str] = {}
    for sample in raw:
        if sample.paper_id not in paper_texts:
            paper_texts[sample.paper_id] = sample.full_text
        if len(paper_texts) >= num_papers:
            break

    console.print(f"[cyan]論文 {len(paper_texts)} 件から合成します[/]")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    by_type: dict[str, int] = {}
    with out_file.open("w", encoding="utf-8") as f:
        for i, (paper_id, text) in enumerate(paper_texts.items(), 1):
            console.print(f"  [cyan]({i}/{len(paper_texts)}) {paper_id[:16]}...[/]")
            try:
                records = synthesize_for_paper(
                    client=client,
                    paper_id=paper_id,
                    full_text=text,
                    n_questions=questions_per_paper,
                    min_spans=min_spans,
                    model=model,
                )
            except Exception as e:
                logger.warning("paper=%s 合成失敗: %s", paper_id, e)
                console.print(f"    [yellow]⚠ skip: {e}[/]")
                continue

            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                by_type[r["query_type"]] = by_type.get(r["query_type"], 0) + 1
                total += 1

            console.print(f"    [green]✓[/] {len(records)} 問 採用")

    dist_str = "  ".join(f"{k}:{v}" for k, v in sorted(by_type.items()))
    console.print(
        f"\n[bold green]✓ 合計 {total} 問 生成 → {out_file}[/]\n"
        f"[dim]クエリ型分布: {dist_str}[/]"
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # 合成は indexing より一段強いモデルを使いたいので、専用 env を優先、
    # なければ classifier を、それもなければ gpt-5 をデフォルトにする。
    _env_synth = os.environ.get(
        "OPENAI_SYNTH_MODEL",
        os.environ.get("OPENAI_CLASSIFIER_MODEL", "gpt-5"),
    )
    # 合成は分類より多段推論を要するので既定を "medium" にする。
    # OPENAI_REASONING_EFFORT で上書き可能。
    _env_reasoning = os.environ.get("OPENAI_REASONING_EFFORT", "medium")

    parser = argparse.ArgumentParser(description="WHY/HOW 質問合成")
    parser.add_argument("--papers", type=int, default=6)
    parser.add_argument("--questions-per-paper", type=int, default=5)
    parser.add_argument("--min-evidence-spans", type=int, default=2)
    parser.add_argument(
        "--generator-model", type=str, default=_env_synth,
        help=f"合成 LLM。既定: .env の OPENAI_SYNTH_MODEL (現在: {_env_synth})。",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default=_env_reasoning,
        choices=["minimal", "low", "medium", "high"],
        help=(
            "gpt-5 系の reasoning_effort。合成は 'medium' 推奨。"
            f"既定: .env の OPENAI_REASONING_EFFORT (現在: {_env_reasoning})。"
        ),
    )
    parser.add_argument("--out", type=str, default="eval_results/why_how_v1.jsonl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
    )

    main(
        num_papers=args.papers,
        questions_per_paper=args.questions_per_paper,
        min_spans=args.min_evidence_spans,
        model=args.generator_model,
        out_path=args.out,
    )
