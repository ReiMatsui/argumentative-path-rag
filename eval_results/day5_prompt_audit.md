# Day 5: インデックス側プロンプト監査レポート

対象: 議論グラフ生成時に LLM に渡しているプロンプトとスキーマ。
目的: 情報欠落（特に WHAT 型で効くファクト）を防ぐ。

## 見つかった問題

### P1. パラフレーズ許可で数値・固有名詞が消える
旧 `NodeItem.text` は「1件のノードとしての短いテキスト」としか説明されておらず、
LLM は自由にパラフレーズしてよいと解釈していた。結果として
`"BLEU improved by +2.1 on WMT14 En-De"` → `"BLEU improved"` のように
モデル名・データセット名・数値が落ちる。
WHAT 型クエリは固有名詞マッチで稼ぐので、この損失が致命的だった。

### P2. EVIDENCE ノードの抽出漏れ
旧プロンプトは EVIDENCE を "empirical data, statistics, or examples" と
曖昧に定義しており、「XX model has 6 layers」「learning rate=3e-4」
のような **スペック型の事実文** が取り漏らされる傾向があった。
WHAT 型はまさにこの層を聞く（"what learning rate…", "how many layers…"）。

### P3. プロンプトの例が汎用ビジネス寄り
"Q3 revenue grew 23%" 等のビジネス例しかなく、NLP 論文ドメインへの
転移が弱かった。few-shot として機能させるなら NLP/ML 例に差し替える必要あり。

### P4. エッジ confidence の偏り
`min_confidence=0.7` でカットしている一方、プロンプトでは "only output
edges with confidence >= 0.7" と指示しているだけで、recall は二重に削られる。
WHY/HOW はエッジを辿るので、ここは recall 寄りにしたい。

## 実装した改善

### a) スキーマを `text` + `source_span` に分離  (`indexing/schemas.py`)

```python
class NodeItem(BaseModel):
    node_type: NodeType
    text: str           # ≦200字の正規化ラベル（パラフレーズ可、表示用）
    source_span: str    # passage の verbatim 部分文字列（消えない本体）
```

- `source_span` は「character-for-character copy」を明示要求。
- `default=""` にしてあるので既存テスト・既存スナップショットは壊れない。
- LLM がパラフレーズしても、原文は `source_span` 側に残る。

### b) `ArgumentNode` に `source_span` と `verbatim_text` を追加  (`models/graph.py`)

```python
source_span: str | None = None

@property
def verbatim_text(self) -> str:
    if self.source_span and self.source_span.strip():
        return self.source_span.strip()
    return self.text
```

以後、**retrieval / context builder / evaluator** はすべて
`verbatim_text` を優先して参照する。これで LLM コンテキスト・
BM25 トークン化・Dense 埋め込み・Evidence-F1 がすべて「原文」基準になる。

### c) NodeClassifier プロンプトを NLP 論文向けに全面書き換え  (`indexing/classifier.py`)

要点:
- ドメインを "scientific literature (especially NLP / ML papers)" に限定。
- EVIDENCE の定義を拡張: *dataset sizes/splits, architecture choices
  (layers, hidden size), hyperparameters (learning rate, batch size,
  epochs), ablation numbers, benchmark scores* を明示列挙。
- Coverage rule を追加: **「抽出漏れより過剰抽出を優先」**。
  "Every sentence that carries factual, definitional, or argumentative
  content should produce at least one node."
- `source_span` について「verbatim substring である」ことを
  プロンプトとスキーマの両方で強制。

### d) `_to_nodes` に verbatim 検証を追加  (`indexing/classifier.py`)

```python
if span and span not in chunk_text:
    logger.warning("chunk_idx=%d: source_span が passage に実在しません …")
    if len(span) < len(text):
        span = ""   # LLM がパラフレーズしたと判断して span を破棄
```

LLM がうっかり改変した span は warning ログ付きで検出し、
不正な場合は空にフォールバックする（=`verbatim_text` が `text` に戻る）。

### e) EdgeExtractor プロンプトも NLP 例に差し替え  (`indexing/extractor.py`)

- 例を Q3 revenue → "BLEU +2.1 on WMT14 En-De", "layer norm",
  "pretraining", "Figure 3 attention weights" に変更。
- `nodes_json` に `text` と `source_span` の両方を渡し、
  "ground your reasoning in source_span to avoid being misled
  by paraphrased labels" というルールを追加。
- recall 寄りルールを追加: "Prefer recall over precision for
  within-same-topic pairs"。

### f) 下流で verbatim を使うように 4 ファイル修正

| ファイル | 変更 |
| -- | -- |
| `retrieval/context_builder.py` | LLM に渡す文脈行を `verbatim_text` に |
| `retrieval/selector.py` (BM25) | BM25 のトークン化対象を `verbatim_text` に |
| `retrieval/embedding_selector.py` | 埋め込みの passage 文字列を `verbatim_text` に |
| `evaluation/evaluator.py` | `retrieved_contexts` の収集を duck-typed に (`getattr(node, "verbatim_text", None) or node.text`) |

※ 最後のだけは Baseline RAG の `_TextNode` が `verbatim_text` を
持たないので getattr 経由にした。

## なぜ WHAT 型の性能が上がると期待できるか

1. **F1 / EM**: WHAT 型は「固有名詞・数値マッチ」が効く。
   `source_span` が原文を保持し、LLM 生成器にもこれを渡すので、
   回答に使う素材が "paraphrase された縮小版" から "原文" に戻る。
2. **Evidence-F1**: 評価側の `retrieved_contexts` が verbatim に
   なることで、gold evidence（QASPER は原文 sentence）との
   token 重なりが直接効く。v1 で AP=0.358 vs Dense=0.461 の差は
   大半がここの「text が paraphrase で縮んでいる」ことに起因する
   可能性が高い。
3. **WHY / HOW**: EVIDENCE の coverage が広がることで、
   因果の起点となるファクトが抜けなくなり、SUPPORTS/DERIVES
   エッジが正しく張れる確率が上がる。

## 改善の範囲・副作用

- **破壊的変更なし**: `source_span` は default="" / None で
  後方互換あり。既存スナップショット・既存テストは通る想定
  （ただし sandbox では pydantic が入っておらず pytest 不可、
  `ast.parse` での構文チェックのみ済）。
- **インデックスは再生成が必要**: 既存の `graph.json`
  スナップショットには `source_span` がない。Day 5 の qasper_mini 再実行で
  自動的に作り直される。
- **トークン量**: プロンプトが ~40% 長くなった。gpt-4o-mini
  換算で 1 ペーパ約 0.3 ¢ → 0.4 ¢。10 ペーパで +1 ¢ 程度。
  コスト見積もり `cost_estimate_day5.md` のレンジに収まる。

## 未着手 (将来課題)

- P4 のエッジ confidence しきい値は今回触っていない。
  Day 5 の再実行結果を見て、WHY/HOW の F1 が低いようなら
  `min_confidence=0.7 → 0.6` に下げる A/B を検討。
- Chunk 境界をまたぐ EVIDENCE→CLAIM エッジは現状の EdgeExtractor
  （チャンク内に閉じている）では取れない。必要なら
  CrossChunkEdgeExtractor を強化する。

## 変更ファイル一覧

- `src/ap_rag/indexing/schemas.py`
- `src/ap_rag/indexing/classifier.py`
- `src/ap_rag/indexing/extractor.py`
- `src/ap_rag/models/graph.py`
- `src/ap_rag/retrieval/context_builder.py`
- `src/ap_rag/retrieval/selector.py`
- `src/ap_rag/retrieval/embedding_selector.py`
- `src/ap_rag/evaluation/evaluator.py`

すべて `python3 -c "import ast; ast.parse(open(...).read())"` で
構文チェック済み。実機動作確認は Day 5 の再実行 (`day5_runbook.md`) で行う。
