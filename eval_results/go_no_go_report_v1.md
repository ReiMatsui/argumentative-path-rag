# Argumentative-Path RAG — 実装前 De-risk 評価レポート (Day 4)

**日付:** 2026-04-20
**目的:** 研究計画書 v6 の手法が「存在論的に立ち上がるか」を、本格的な論文執筆に入る前に最小実験で確認する。
**判定:** **CONDITIONAL NO-GO** — 現在の定式化のまま QASPER を主ベンチマークにするのは危険。本文書はその根拠と推奨ピボットを示す。

---

## 1. 実施したこと

### 1.1 実装（Day 1）

- 研究計画書 v6 §4.3「長距離議論依存の接続」の中心指標である **Evidence-F1** を実装（QASPER 公式の gold evidence スパンと取得文脈の token-F1）。
- OpenAI Embeddings (`text-embedding-3-small`, 1536 dim) を `SentenceTransformer` 互換アダプタとして AP-RAG 入口選定 / CrossChunk / Dense ベースラインに共有注入できるように改修（M1 MacBook で torch を避けるため）。
- 比較ベースライン BM25RAG / DenseRAG と ArgumentativeRAG を同一 QA サンプルで並列評価する実行器 (`scripts/qasper_mini.py`) を整備。EM / F1 / Evidence-F1 / Faithfulness / Hallucination Rate / Answer Correctness を並列計測。

### 1.2 評価ラン (Day 2, qasper_main_v1)

| 項目 | 値 |
| --- | --- |
| ベンチマーク | allenai/qasper (validation) |
| 論文数 | 3 |
| 質問数/論文 | 最大 5 |
| 実質サンプル数 | **N = 8** |
| 埋め込み | `text-embedding-3-small` (OpenAI) |
| 生成 LLM | `gpt-4o-mini` |
| LLM-as-Judge | 有効 (gpt-4o-mini, faithfulness / hallucination / correctness) |
| CrossChunk | **OFF** |

---

## 2. 観測された結果 (N=8)

| 指標 | AP-RAG | BM25 | Dense | 優位 |
| --- | ---: | ---: | ---: | --- |
| EM | 0.000 | 0.000 | 0.000 | — |
| **F1** | 0.067 | 0.169 | **0.171** | Dense ≈ BM25 > **AP** |
| **Evidence-F1** | 0.358 | 0.385 | **0.461** | Dense > BM25 > **AP** |
| **Answer Correctness** | 0.475 | 0.500 | **0.675** | Dense > BM25 > **AP** |
| Faithfulness | 0.938 | 1.000 | 1.000 | BM25 = Dense > AP |
| **Hallucination Rate↓** | **0.000** | 0.125 | 0.125 | **AP** > BM25 = Dense |

### 2.1 何が起きているか

**AP-RAG は精度に関するすべての主要指標で負けている。** 唯一上回っているのは「余計なことを言わない (hallucination=0%)」だけで、それは単に「正しい答えを拾えなかったとき素直に『わからない』と言っている」結果に近い可能性が高い（Correctness 0.475 < 0.5 はそれを裏付ける）。

**Evidence-F1 で Dense に 10 ポイント以上負けている**のが特に重い。この指標は研究計画書 v6 §4.3 で「新規貢献②: 長距離議論依存の接続」の中心指標として位置付けたものだ。ここで負けていると、論文の中核の主張が直接揺らぐ。

### 2.2 クエリ型分布の致命的な偏り

LLM 分類器 (`QueryClassifier`) の出力:

```
WHAT: 8
WHY: 0   HOW: 0   EVIDENCE: 0   ASSUMPTION: 0
```

**研究計画書が想定していた WHY / HOW / EVIDENCE クエリは、QASPER validation の上位論文からは 1 件も出てこなかった。** 3 論文 × 5 問で N=8（15 問上限に対し）という少なさは、QASPER では多くの論文の QA ペア数がそもそも少ないことを反映している。5 papers × 10 questions に増やしても実質 13 問で、やはり WHAT:12 HOW:1 の偏りだった（index 実行が sandbox の process lifetime 制限で完走しなかったため v2 JSON は得られず、ロードログのみ確認）。

**§4.5「クエリ型別分析：保険としての位置付け」という計画は、QASPER をメインにする限り成立しない。** WHY/HOW/EVIDENCE クエリが評価サンプルにほぼ存在しないため、適応戦略の効果を示す舞台がない。

---

## 3. 根本的な診断

### 3.1 WHAT クエリで勝てない理由の仮説

WHAT クエリは「何ですか」「どれですか」式の事実検索で、keyword / embedding 類似度で十分対応できる。ここで AP-RAG が負けるのは当然に近い:

1. **AP-RAG は入口選定 → グラフ探索 → コンテキスト構築と、チャンク直接取得より工程が多い**。各工程で多少のノイズが混入し、最終的に渡す文脈が「正解を含むスパン」から遠ざかりやすい。
2. **議論グラフのノード (CLAIM/EVIDENCE/ASSUMPTION/…) は文のパラフレーズ**である。元の本文の正解スパンそのものが消えている可能性がある（Evidence-F1 が token-F1 なので、パラフレーズによってトークン一致率が下がる）。
3. **`max_nodes=15` のコンテキスト予算は AP-RAG の BEST 状態前提**。WHAT クエリでは拡張せず top-5 チャンクだけ渡す Dense のほうがシグナル:ノイズ比が良い。

### 3.2 メソッドの強みが立証できない理由

研究計画書 v6 §4.5 の想定:
- WHY ではグラフ (CLAIM←SUPPORTS←EVIDENCE) が効く
- HOW では順方向 DERIVES 鎖が効く
- EVIDENCE では直接エッジが効く

しかし QASPER validation セットでは WHY / HOW / EVIDENCE は**出現しない**。想定と評価用データセットがミスマッチ。

---

## 4. 判定: CONDITIONAL NO-GO

**いま GO すると何が起きるか:**
- 本格評価で N=50〜200 に拡大しても、QASPER のクエリ型分布は同じように WHAT 偏重になる公算が高い。
- そのまま論文を書くと「我々の手法は BM25 と Dense に F1 / Evidence-F1 / Correctness すべてで負ける」という結果になり、§4.5 の保険（クエリ型別勝敗分解）も WHY/HOW/EVIDENCE サンプル欠如で成立しない。
- 中核主張（§3 の新規貢献②: 長距離議論依存の接続）が Evidence-F1 で否定されるため、クエリ型適応（新規貢献①）だけが残る。しかし適応が効く舞台（QASPER）で適応が効く型（WHY/HOW）が存在しないため、これも立証できない。

**この状態の研究は主張できる貢献が CrossChunk アブレーションでの「標準 RAG を何 % 下回らない」という守備的な話に縮小する。修士論文として厳しい。**

---

## 5. 推奨ピボット（優先順）

### Option A (強く推奨): ベンチマーク変更

研究計画書 v6 §4.3 の「ベンチマーク」セクションで QASPER を「必須」としていたのを再考する。

- **HotpotQA / MuSiQue**: マルチホップ QA。WHY / HOW / EVIDENCE の自然な登場率が高い。AP-RAG のグラフ探索が効く舞台。
- **2WikiMultihopQA**: 反事実・前提 (ASSUMPTION) を問う設問が含まれる。
- **QASPER の fiction of fit**: 計画書は QASPER を「科学論文長文書QA」として採用したが、QASPER QA ペアの多くは事実問合せ (WHAT) で、WHY/HOW のクエリ密度が想定より低い。

HotpotQA / MuSiQue に主軸を移し、QASPER を「単一文書内の長距離依存を見る補助ベンチマーク」に格下げするのが最も低コストで自然。

### Option B: 入口選定と文脈予算のチューニング

もし QASPER を残すなら:
- `max_nodes` を 15 → **5〜8** に絞る (Dense と等条件にする)
- WHAT 型に対しては **グラフ探索を実質 no-op** にして入口チャンクそのものを top-k で返す "WHAT fallback" を実装
- この状態で再計測して、WHAT での劣化を止められるか見る

これは「WHAT では Dense とタイ」まで持っていくための守備策。新規貢献の積極主張にはならない。

### Option C (非推奨): N を増やして信号が変わることに賭ける

N=8 → N=50 に増やしても分布は WHAT 偏重のまま。AP-RAG が Dense を上回る確率は低い。時間と $ を浪費する可能性が高い。

---

## 6. 次の一手 (最低コストの検証)

**1 日で実行できる検証 (Option A の生存確認):**

1. `ap_rag.evaluation.benchmarks` に `HotpotQA` / `MuSiQue` ローダを追加する（公開 dataset なので実装は 2〜3 時間）。
2. 現 `qasper_mini.py` と同じ 3 ベースライン × 判定器で N=20 程度を回す。
3. 期待: WHY/HOW 率が 30〜50% まで上がり、AP-RAG が少なくとも F1 で同点以上に持ち直す。
4. その結果次第で本格的な Day 5 以降の計画を引き直す。

**所要:** 実装 3h + ラン 1h + 分析 1h ≈ **5h / $5 以下**

これを通してから論文執筆に入るのが安全。

---

## 7. v1 実験の生データ

参照: `eval_results/qasper_main_v1.json`

完全なメトリクスは上表参照。per-sample dump は v2 ランで取得予定だったが、sandbox の process lifetime 制限（バックグラウンド Python が ~30-45 秒で kill される）により今回は取得できなかった。実機 (M1 MacBook) で `scripts/qasper_mini.py --papers 3 --questions 5 --embedding-backend openai --save-json eval_results/qasper_main_v1_persample.json` を走らせれば per-sample 診断データが取れる（per_sample 機能は既に `src/ap_rag/evaluation/metrics.py` / `evaluator.py` / `scripts/qasper_mini.py` に組み込み済み）。

---

## 8. 要約

- **v1 実験 (N=8, 全 WHAT)** で AP-RAG は F1 / Evidence-F1 / Correctness のすべてでベースラインに負けた。Hallucination だけ勝ち。
- 負けの主因は (a) クエリ型分布が WHAT 一色で方法の強みが出る場面がない、(b) グラフ経由のコンテキスト構築が WHAT で純粋にオーバーヘッドになっている、の 2 点。
- このまま QASPER で論文を書き通すと、新規貢献①・②ともに立証できない。
- 推奨: ベンチマークを HotpotQA / MuSiQue に寄せて再判定する (Day 5 で 5h 以内に可能)。
