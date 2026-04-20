# Day 5 実行手順 (M1 MacBook 用ランブック)

**目的:** (1) v1 と同条件でモデル格上げ再実行、(2) WHY/HOW 合成セットで AP-RAG の真価を確認。

---

## 前提

- `.env` に `OPENAI_API_KEY` を設定済み。
- `uv sync` 済み (Day 1〜2 で完了)。
- 本セッションで以下のコード変更が入っている:
  - `scripts/qasper_mini.py` に `--classifier-model` / `--generator-model` / `--judge-model` CLI フラグ追加。indexing も SetFit ではなく CLI フラグで切替可能に。
  - `scripts/synthesize_why_how.py` 新規 — QASPER 本文から WHY/HOW 質問を LLM 合成。
  - `scripts/why_how_eval.py` 新規 — 合成 JSONL を消費して AP / BM25 / Dense を同ハーネスで評価。

---

## 1. モデル格上げ版 v1 再実行 (`qasper_main_v2_gpt5mini`)

**狙い:** v1 と同じ 3 papers × 5 Q で、全モデルを `gpt-5-mini` に揃え、indexing 品質が上がるだけで WHAT 型の劣勢が縮まるかを見る。

```bash
cd ~/argumentative-path-rag
uv run python scripts/qasper_mini.py \
  --papers 3 --questions 5 \
  --embedding-backend openai \
  --classifier-model gpt-5-mini \
  --generator-model gpt-5-mini \
  --judge-model gpt-5-mini \
  --save-json eval_results/qasper_main_v2_gpt5mini.json \
  2>&1 | tee eval_results/qasper_main_v2_gpt5mini.log
```

想定コスト: **~$0.26** (`cost_estimate_day5.md` §3 の B 行)
想定実行時間: M1 CPU で 6〜10 分 (indexing が律速; 並列化されているが gpt-5-mini は gpt-4o-mini より少し遅い)

判定基準 (v1 比):
- AP-RAG の Evidence-F1 が 0.358 → **0.46 以上** に乗れば、indexing 品質で Dense に追いついたと見なす。
- AP-RAG の F1 (WHAT 型) が 0.067 → **0.15 以上**に乗れば、グラフ経由でも最低限生き残ることが示される。
- 動かなければ classifier だけ gpt-5 (full) に上げて再実行 (§1b)。

### 1b. classifier のみ gpt-5 に格上げ (必要時のみ)

```bash
uv run python scripts/qasper_mini.py \
  --papers 3 --questions 5 \
  --embedding-backend openai \
  --classifier-model gpt-5 \
  --generator-model gpt-5-mini \
  --judge-model gpt-5-mini \
  --save-json eval_results/qasper_main_v2_gpt5classifier.json \
  2>&1 | tee eval_results/qasper_main_v2_gpt5classifier.log
```

想定コスト: **~$0.44**

---

## 2. WHY/HOW 合成セット作成 + 評価

### 2.1 合成 (`why_how_v1.jsonl`)

**狙い:** QASPER validation には WHY/HOW がほぼ存在しない。AP-RAG の強み (CLAIM ← SUPPORTS ← EVIDENCE, DERIVES 鎖など) が効く型を人工的に用意し、実際に効くか見る。

```bash
cd ~/argumentative-path-rag
uv run python scripts/synthesize_why_how.py \
  --papers 6 --questions-per-paper 5 \
  --min-evidence-spans 2 \
  --generator-model gpt-5 \
  --out eval_results/why_how_v1.jsonl \
  2>&1 | tee eval_results/why_how_v1.log
```

想定コスト: **~$0.15** (gpt-5 で 30 問程度)
想定実行時間: 3〜6 分

**生成後の手動チェック必須:**
- `why_how_v1.jsonl` を目視で 5〜10 問レビュー。
- 各レコードの `evidence` が本文に実在する短いスパンか。
- `question` が本当に WHY/HOW を問うか (表層だけ Why で始まっているが実は WHAT な質問を排除)。
- 不適切なら手で 1 行ずつ削除して OK。

### 2.2 評価 (`why_how_main_v1.json`)

```bash
uv run python scripts/why_how_eval.py \
  --in eval_results/why_how_v1.jsonl \
  --embedding-backend openai \
  --classifier-model gpt-5-mini \
  --generator-model gpt-5-mini \
  --judge-model gpt-5-mini \
  --save-json eval_results/why_how_main_v1.json \
  2>&1 | tee eval_results/why_how_main_v1.log
```

想定コスト: **~$0.50**
想定実行時間: 10〜18 分

**判定基準 (研究テーマ続行可否):**
1. **AP-RAG の F1 が BM25 / Dense を上回る** → 研究テーマ続行 GO。WHY/HOW に強いという主張が数値で裏付けられる。
2. **F1 はタイだが Evidence-F1 が AP-RAG > Dense** → 継続 GO。新規貢献②「長距離議論依存の接続」の根拠となる。
3. **どちらも負け or タイ** → NO-GO。ベンチマーク / 手法の再設計が必要 (HotpotQA 系への pivot も視野)。

---

## 3. 実行後の報告テンプレ

ラン完了後、以下 3 ファイルを私に渡してください:

1. `eval_results/qasper_main_v2_gpt5mini.json` (または `qasper_main_v2_gpt5classifier.json`)
2. `eval_results/why_how_v1.jsonl` (生成された合成セット)
3. `eval_results/why_how_main_v1.json`

これを元に Day 5 GO/NO-GO 判定を更新します。

---

## 4. 想定総コスト (Day 5 合計)

| 項目 | 想定コスト |
| --- | ---: |
| v1 再実行 (gpt-5-mini 統一) | $0.26 |
| WHY/HOW 合成 (gpt-5) | $0.15 |
| WHY/HOW 評価 (gpt-5-mini) | $0.50 |
| 予備リラン (1 回) | $0.80 |
| **合計** | **~$1.71** |

残予算に対して十分セーフ。

---

## 5. トラブルシュート

**Q. `pydantic` の `SynthBatch` が parse 失敗する**
→ `--generator-model` を `gpt-5` か `gpt-4.1` に変える (構造化出力の精度が上がる)。`gpt-5-mini` では JSON が崩れる可能性がある。

**Q. evidence スパン verify が軒並み失敗して 0 問になる**
→ LLM が evidence をパラフレーズしている。`--min-evidence-spans 1` に緩める + 事後に手動でチェック。

**Q. qasper_main_v2 の F1 が v1 とほぼ同じ**
→ classifier を gpt-5 (full) に上げた版 (§1b) を回す。それでも変わらなければ「WHAT 型に対する AP-RAG のアーキ的不利は model 品質では埋まらない」と結論。
