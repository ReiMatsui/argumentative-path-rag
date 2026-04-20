# Day 5 コスト見積もり — モデル格上げ + WHY/HOW 検証

**日付:** 2026-04-20
**対象:** v1 (3 papers × 5 Q, 実質 N=8) と同規模での再実行コスト、および WHY/HOW 合成セット (N≒30) の評価コスト。

---

## 1. v1 ランで発生している LLM 呼び出しの内訳

パイプラインのコードから逆算した、v1 (3 papers × 5 Q, realized N=8) の呼び出し数と 1 回あたりの目安トークン。

| フェーズ | モデル種別 | 呼び出し数 | 入力 (平均) | 出力 (平均) | 入力合計 | 出力合計 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NodeClassifier (chunk → nodes) | classifier | 75 chunks | 500 | 100 | 37,500 | 7,500 |
| EdgeExtractor (chunk nodes → edges) | classifier | 75 | 800 | 200 | 60,000 | 15,000 |
| QueryClassifier | generator | 8 | 200 | 5 | 1,600 | 40 |
| AnswerGenerator AP-RAG | generator | 8 | 2,500 | 150 | 20,000 | 1,200 |
| AnswerGenerator BM25 | generator | 8 | 1,500 | 150 | 12,000 | 1,200 |
| AnswerGenerator Dense | generator | 8 | 1,500 | 150 | 12,000 | 1,200 |
| LLMJudge (faith + halluc + correct) | generator | 3 × 3 × 8 = 72 | 600 | 10 | 43,200 | 720 |
| Embedding (text-embedding-3-small) | embedding | — | — | — | ~25,000 | — |

**合計:**
- Classifier 模型: **97.5k in / 22.5k out**
- Generator 模型: **88.8k in / 4.36k out**
- Embedding: 25k (cost negligible)

注: チャンク数 75 は「25 chunks/paper × 3」の概算。QASPER 論文本体の長さに依存。LLM 分類器/抽出器は 1 チャンク 1 コールでリクエストする構造。

---

## 2. モデル別 単価 (1M tokens, 2026-04 時点)

| モデル | Input | Output | 備考 |
| --- | ---: | ---: | --- |
| gpt-4o-mini | $0.15 | $0.60 | v1 の generator |
| gpt-4o | $2.50 | $10.00 | v1 の classifier |
| gpt-4.1 | $2.00 | $8.00 | 4o のリプレース世代 |
| gpt-4.1-mini | $0.40 | $1.60 | mini tier |
| **gpt-5-mini** | **$0.75** | **$4.50** | 汎用 mini、品質は 4o-mini を大きく上回る |
| **gpt-5** | **$1.25** | **$10.00** | flagship (reasoning 含) |

---

## 3. v1 再実行コスト (1 ランあたり)

**重要: v1 実行時、`qasper_mini.py` は classifier も generator も `gpt-4o-mini` をハードコードしていた** (config.py の `openai_classifier_model=gpt-4o` は Settings 経由のパスでしか使われない)。したがって v1 は「全部 gpt-4o-mini」で動いていた。これは「indexing の質が低かったから AP-RAG のグラフ生成が弱くて負けた」という仮説を強く支持する。

| 構成 | Classifier コスト | Generator コスト | **合計** |
| --- | ---: | ---: | ---: |
| **v1 実測 (All gpt-4o-mini)** | $0.028 | $0.016 | **~$0.04** |
| 参考: 旧計画 (gpt-4o + 4o-mini) | $0.469 | $0.016 | ~$0.49 |
| **B. All gpt-5-mini** (推奨) | $0.174 | $0.086 | **$0.26** |
| **C. gpt-5 + gpt-5-mini** | $0.347 | $0.086 | **$0.44** |
| **D. All gpt-5** (最強) | $0.347 | $0.155 | **$0.50** |
| E. gpt-4.1 + gpt-4.1-mini | $0.375 | $0.043 | $0.42 |

**推奨: B → C の順で検証。**
理由:
1. v1 は gpt-4o-mini だけで動いており、特に「CLAIM / EVIDENCE / ASSUMPTION 等の構造化ノード分類」と「SUPPORTS / DERIVES 等の因果エッジ抽出」は 4o-mini では弱い可能性が高い。**gpt-5-mini へ上げるだけで indexing 品質は大きく改善する見込み**。
2. B 構成でも WHAT 劣勢が解消しない場合、C で classifier を gpt-5 に格上げし「グラフ品質の上限」を見る。
3. D は最強だが +$0.06/run にしかならないので、B で満足いかなければ直接 C/D に飛ぶ判断でよい。
4. 元コード (`config.py`) の `gpt-4o` は `qasper_mini.py` 側のハードコードでオーバーライドされていたため、**v1 ランのデータは「良い classifier」を使った比較にはなっていない** 点を明記。

---

## 4. Day 5 全体 (v1 再検証 + WHY/HOW 検証) コスト

### 4.1 WHY/HOW 合成データセット構築コスト

30 問を QASPER 論文本文から LLM で生成 (質問 + gold evidence スパン)。

| モデル | per Q (in/out) | 合計 in | 合計 out | コスト |
| --- | --- | ---: | ---: | ---: |
| gpt-5-mini | 1,500 / 300 | 45k | 9k | **$0.075** |
| gpt-5 | 1,500 / 300 | 45k | 9k | $0.146 |

WHY/HOW は推論が必要なので gpt-5 の採用を推奨 ($0.15 で全体の誤差範囲)。

### 4.2 WHY/HOW 評価ラン (N≒30)

v1 と同じ構成を Q 数だけスケール (3.75x で generator 側が膨らむ、classifier 側は同じ論文なのでそのまま)。

| 構成 | Classifier | Generator × 3.75 | **合計** |
| --- | ---: | ---: | ---: |
| B. All gpt-5-mini | $0.174 | $0.323 | **$0.50** |
| C. gpt-5 + gpt-5-mini | $0.347 | $0.323 | **$0.67** |

### 4.3 Day 5 合計

| 項目 | B で統一 | C で統一 |
| --- | ---: | ---: |
| v1 再実行 | $0.26 | $0.44 |
| WHY/HOW 合成 (gpt-5 推奨) | $0.15 | $0.15 |
| WHY/HOW 評価 | $0.50 | $0.67 |
| リラン予備 (×1 分) | $0.80 | $1.15 |
| **Day 5 合計見積もり** | **~$1.71** | **~$2.41** |

**残予算 ($50 → v1 で ~$0.5 消費済み) に対して十分セーフ。**

---

## 5. スケールアップ時の参考値 (本格評価)

論文 10 × 1 論文あたり 10 Q (N≒70〜100 想定) に拡張した場合の 1 ランコスト:

| 構成 | 推定コスト |
| --- | ---: |
| B. All gpt-5-mini | ~$2.5 |
| C. gpt-5 + gpt-5-mini | ~$4.0 |

**$50 枠であれば 10 回ラン (ablation 込み) が現実的。**

---

## 6. 実装上の切替コスト

`.env` の 2 行を書き換えるだけで切替可能:

```bash
# 従来
OPENAI_CLASSIFIER_MODEL=gpt-4o
OPENAI_GENERATOR_MODEL=gpt-4o-mini

# 推奨 (B 構成)
OPENAI_CLASSIFIER_MODEL=gpt-5-mini
OPENAI_GENERATOR_MODEL=gpt-5-mini
```

ただし以下のモジュールで `"gpt-4o-mini"` がデフォルト値としてハードコードされており、`qasper_mini.py` が settings を通さず直接インスタンス化している箇所があれば `--model` 系フラグの追加が必要:

- `src/ap_rag/evaluation/baselines.py` (`BM25RAG`, `DenseRAG`): `model="gpt-4o-mini"` 既定 → 呼び出し側で上書きされているか要確認
- `src/ap_rag/evaluation/metrics.py` (`LLMJudge`): `model="gpt-4o-mini"` 既定 → 同上
- `src/ap_rag/generation/generator.py` (`AnswerGenerator`): `model="gpt-4o-mini"` 既定

**次の実装タスク:** `qasper_mini.py` に `--generator-model` / `--classifier-model` / `--judge-model` の CLI フラグを追加し、Settings と同時に 3 箇所 (AP-RAG の generator, baselines の generator, judge) に反映させる。

---

## 7. まとめ

1. v1 ランの実コストは ~$0.49。今使っている gpt-4o + gpt-4o-mini 構成は、classifier 側 (indexing) が gpt-4o で 95% のコスト比率を占めている。
2. **gpt-5-mini は gpt-4o-mini より能力が上で、gpt-4o より安い。** v1 と同じ論文数・質問数で再実行すると `B: All gpt-5-mini` で **$0.26 (旧比 -47%)**。
3. Day 5 の 2 実験 (v1 再実行 + WHY/HOW 評価) 全部で **~$1.7〜2.4**。予算内。
4. 実装変更は .env 2 行 + qasper_mini.py の CLI 拡張で済む。
