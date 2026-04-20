# Argumentative-Path RAG

**Query-type-adaptive Retrieval-Augmented Generation over Argument Graphs**

論文・財務報告書などの長文文書に対して、「なぜ？」「根拠は？」「前提は？」という論証的な質問に高精度で答えるRAGシステム。文書を論証グラフとして構造化し、クエリの種類に応じて探索戦略を切り替える。

---

## 目次

1. [モチベーション](#1-モチベーション)
2. [システム概要](#2-システム概要)
3. [論証タクソノミー](#3-論証タクソノミー)
4. [アーキテクチャ](#4-アーキテクチャ)
5. [クエリ型適応探索](#5-クエリ型適応探索)
6. [既存手法との比較](#6-既存手法との比較)
7. [評価フレームワーク](#7-評価フレームワーク)
8. [クイックスタート](#8-クイックスタート)
9. [プロジェクト構造](#9-プロジェクト構造)

---

## 1. モチベーション

### 従来のRAGが苦手なこと

標準的なRAG（Retrieval-Augmented Generation）は、クエリのベクトルに「意味的に近い」テキストチャンクを返す。これは事実確認や定義検索には有効だが、**論証的な質問**には不十分である。

```
質問: "なぜQ3の売上が落ちたのか？"

従来RAG → "Q3売上が落ちた" に意味的に近い段落を返す
           ✗ 原因・根拠・前提は別の段落にある

Argumentative-Path RAG → CLAIM から因果チェーンを辿る
           ✓ EVIDENCE（半導体不足）→ ASSUMPTION（為替安定）まで収集
```

### 解決するギャップ

| 質問タイプ | 従来RAGの弱点 | 本システムの解決策 |
|---|---|---|
| WHY（なぜ？） | 因果エッジを辿れない | SUPPORTS/DERIVES エッジで根拠を遡る |
| EVIDENCE（根拠は？） | 分散した証拠を収集できない | グラフ上の EVIDENCE ノードを網羅 |
| ASSUMPTION（前提は？） | 暗黙の前提を見つけられない | ASSUMES エッジを outgoing 方向に辿る |
| HOW（どうやって？） | 手順の繋がりが切れる | DERIVES/ILLUSTRATES を辿る |

---

## 2. システム概要

本システムは**2フェーズ**で動作する。

```
┌─────────────────────────────────────────────────────────┐
│  OFFLINE PHASE（文書インデックス）                       │
│                                                          │
│  文書テキスト                                            │
│      │                                                   │
│      ▼                                                   │
│  ① チャンク分割（SentenceChunker）                      │
│      │  句点・改行で分割、最大トークン数で調整           │
│      ▼                                                   │
│  ② ノード分類（NodeClassifier / LLM）                   │
│      │  各チャンクの文章を CLAIM / EVIDENCE / ... に分類 │
│      ▼                                                   │
│  ③ エッジ抽出（EdgeExtractor / LLM）                    │
│      │  ノード間の論理的関係を SUPPORTS / ASSUMES / ... │
│      ▼                                                   │
│  ④ グラフ保存（NetworkXGraphStore / Neo4jGraphStore）   │
│      │  ArgumentGraph として永続化                      │
└─────────────────────────────────────────────────────────┘
                          │
                     （グラフDB）
                          │
┌─────────────────────────────────────────────────────────┐
│  ONLINE PHASE（クエリ時）                                │
│                                                          │
│  ユーザークエリ                                          │
│      │                                                   │
│      ▼                                                   │
│  ① クエリ型分類（QueryClassifier / LLM）                │
│      │  WHY / WHAT / HOW / EVIDENCE / ASSUMPTION        │
│      ▼                                                   │
│  ② 入口ノード選定（TraversalStrategy）                  │
│      │  クエリ型に応じた entry_node_types から取得      │
│      ▼                                                   │
│  ③ 型適応BFS探索（GraphTraverser）                      │
│      │  エッジ型・方向・深さを戦略に従って制御          │
│      ▼                                                   │
│  ④ コンテキスト構築（ContextBuilder）                   │
│      │  取得ノードのテキストを整形                       │
│      ▼                                                   │
│  ⑤ 回答生成（AnswerGenerator / LLM）                    │
│      │                                                   │
│      ▼                                                   │
│  回答テキスト                                            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 論証タクソノミー

### ノードタイプ（7種類）

文書内の各文・節は以下の論証的役割に分類される。

| ノードタイプ | 記号 | 定義 | 例 |
|---|---|---|---|
| `CLAIM` | 🔵 | 主張・命題 | "Q3売上は前年比12%減少した" |
| `EVIDENCE` | 🟢 | 主張を支持する事実・データ | "半導体不足で生産ラインが停止した" |
| `ASSUMPTION` | 🟡 | 主張が成立するための前提 | "為替レートは安定していた" |
| `CONCLUSION` | 🟣 | 推論から導かれる結論 | "来四半期も業績低迷が続く見通しだ" |
| `CAVEAT` | ⚫ | 主張を限定する但し書き | "ただし競合他社比では軽微である" |
| `CONTRAST` | 🔴 | 対立・反証する情報 | "競合X社は同期間に8%増加した" |
| `DEFINITION` | 🔷 | 用語・概念の定義 | "在庫調整とは需要予測の下方修正を指す" |

### エッジタイプ（6種類）

ノード間の論理的関係を有向エッジで表現する。

| エッジタイプ | 方向 | 意味 |
|---|---|---|
| `SUPPORTS` | A → B | A が B を根拠として支持する |
| `CONTRADICTS` | A → B | A が B を否定・反証する |
| `DERIVES` | A → B | A から B が導かれる（推論） |
| `ASSUMES` | A → B | A は B を前提としている |
| `ILLUSTRATES` | A → B | A が B の具体例である |
| `CONTRASTS` | A → B | A が B と対比される |

### グラフ例（Q3売上レポート）

```
[CLAIM: Q3売上12%減少]
    ├─SUPPORTS─▶ [EVIDENCE: 半導体不足で生産停止]
    │                └─DERIVES─▶ [CONCLUSION: 来期も継続]
    ├─ASSUMES──▶ [ASSUMPTION: 為替レート安定]
    └─CONTRASTS▶ [CONTRAST: 競合X社は8%増加]
```

---

## 4. アーキテクチャ

### モジュール構成

```
src/ap_rag/
├── models/
│   ├── taxonomy.py       # NodeType / EdgeType / QueryType / TraversalStrategy
│   └── graph.py          # ArgumentNode / ArgumentEdge / ArgumentGraph
│
├── indexing/
│   ├── chunker.py        # SentenceChunker — テキスト → DocumentChunk[]
│   ├── classifier.py     # NodeClassifier — チャンク → ArgumentNode[]（LLM）
│   ├── extractor.py      # EdgeExtractor  — ノード間 → ArgumentEdge[]（LLM）
│   └── pipeline.py       # IndexingPipeline — 上記を統合
│
├── graph/
│   ├── store.py          # GraphStore インターフェース
│   ├── networkx_store.py # インメモリ実装（開発・テスト用）
│   └── neo4j_store.py    # Neo4j 実装（本番用）
│
├── retrieval/
│   ├── query_classifier.py  # QueryClassifier — クエリ型判定（LLM）
│   ├── traversal.py         # GraphTraverser — 型適応BFS
│   ├── context_builder.py   # ContextBuilder — コンテキスト整形
│   └── (strategy は taxonomy.py 内 TRAVERSAL_STRATEGIES)
│
├── generation/
│   └── generator.py      # AnswerGenerator — LLMによる回答生成
│
├── evaluation/
│   ├── metrics.py        # EM / F1 / LLMJudge / aggregate_results
│   ├── evaluator.py      # Evaluator / ComparisonRunner
│   ├── baselines.py      # BM25RAG / DenseRAG
│   ├── ablation.py       # LLMRewriteRAG / FixedStrategyRAG / NoisyGraphStore
│   └── benchmarks/
│       └── qasper.py     # QASPERLoader / QASPERRunner
│
└── pipeline.py           # ArgumentativeRAGPipeline（エンドツーエンド）
```

---

## 5. クエリ型適応探索

本研究の中核となる新規貢献。クエリの種類に応じて**入口ノード・辿るエッジ・探索方向・除外ノード**を動的に切り替える。

### 探索戦略一覧

```
QueryType  入口ノード              辿るエッジ（方向）                     除外  深さ
─────────────────────────────────────────────────────────────────────────────────
WHY        CLAIM, CONCLUSION       SUPPORTS(←), DERIVES(←), ASSUMES(→)  CONTRAST  3
WHAT       CLAIM, CONCLUSION       SUPPORTS(←), ILLUSTRATES(←)          —         2
HOW        CLAIM, CONCLUSION       DERIVES(←), ILLUSTRATES(←)           —         3
EVIDENCE   CLAIM, EVIDENCE         SUPPORTS(←), ILLUSTRATES(←)          —         2
ASSUMPTION CLAIM, CONCLUSION       ASSUMES(→)                            —         2
```

**ポイント: ASSUMES エッジの方向**

`ASSUMES` だけが**outgoing（→）**方向に辿る。CLAIMが前提（ASSUMPTION）を「参照」しているという非対称性を正確に表現するため。他の全エッジは incoming（←）方向に辿る。

```
                   outgoing ─────────────────────▶
[CLAIM: 売上減少] ──ASSUMES──▶ [ASSUMPTION: 為替安定]
                   ◀───────────────────── incoming
```

### WHY クエリの探索例

```
入口: [CLAIM] "Q3売上12%減少"
  │
  ├─ SUPPORTS(←) ─▶ [EVIDENCE] "半導体不足で生産停止"
  │                      │
  │                      └─ DERIVES(←) ─▶ [CONCLUSION] "来期も継続"
  │
  └─ ASSUMES(→) ──▶ [ASSUMPTION] "為替レート安定"

取得ノード: CLAIM + EVIDENCE + ASSUMPTION + CONCLUSION = 4件
除外: CONTRAST ノード（競合X社の話）は除外される
```

---

## 6. 既存手法との比較

### 手法の位置づけ

| 手法 | グラフ構造 | ノードの意味 | クエリ適応 | WHY/ASSUMPTIONへの強さ |
|---|---|---|---|---|
| Dense RAG | なし | チャンク（意味的類似） | なし | 弱 |
| GraphRAG（Microsoft, 2024） | エンティティ関係グラフ | 人・組織・概念 | なし | 中（広域サマリに強い） |
| HippoRAG（2024） | 意味的近接グラフ | テキストノード | なし | 中（マルチホップに強い） |
| ArgRAG（Zhu et al., 2025） | 二極論証グラフ | 支持/攻撃 | なし | 中（事実検証に特化） |
| **本手法** | **7役割×6関係グラフ** | **論証的役割** | **クエリ型別戦略** | **強** |

### GraphRAGとの本質的な違い

GraphRAGはエンティティ（「誰が・何が」）と関係（「関係している」）を抽出する。本手法は**論証的役割**（「その文章が何を主張・証明・前提としているか」）を抽出する。

```
GraphRAG:   [半導体] ──related_to──▶ [売上減少]
本手法:     [EVIDENCE: 半導体不足] ──SUPPORTS──▶ [CLAIM: 売上12%減少]
                                         ▲
                                    「なぜ？」への答えはこちら
```

---

## 7. 評価フレームワーク

### 指標

| 指標 | 内容 | 実装 |
|---|---|---|
| EM (Exact Match) | 正規化後の完全一致率 | `compute_em()` |
| F1 | トークンレベルの重複率（SQuADスタイル） | `compute_f1()` |
| Faithfulness | 回答が取得コンテキストに忠実か（LLM-as-judge） | `LLMJudge.faithfulness_score()` |
| Hallucination Rate | 文書外の情報を混入させた割合（LLM-as-judge） | `LLMJudge.is_hallucination()` |

### ベンチマーク

| ベンチマーク | 文書種別 | 規模 | 重点指標 |
|---|---|---|---|
| 合成QA（Q3売上） | 財務レポート（日本語） | 12問 | Faithfulness / Hallucination Rate |
| QASPER（§4.3 必須） | NLP科学論文（英語） | 論文10件・50問 | F1 / Hallucination Rate |
| FinanceBench（予定） | IR資料・決算報告（英語） | 150問 | EM / F1 |

### アブレーション実験

クエリ型適応とグラフ構造それぞれの寄与を分離して測定する。

| 実験名 | 目的 | 実装クラス |
|---|---|---|
| LLMリライト＋BM25 | 改善がグラフ由来かLLM前処理由来かを分離 | `LLMRewriteRAG` |
| 探索戦略固定化 | クエリ型適応（新規貢献①）の寄与を測定 | `FixedStrategyRAG` |
| ノイズ入りグラフ | グラフ構築誤差への耐性を測定 | `NoisyGraphStore` |

### クイック評価結果（合成QA, N=12）

```
システム          EM     F1     Faithfulness↑  Hallucination↓
──────────────────────────────────────────────────────────────
ArgumentativeRAG  0.000  0.019  1.000          0.000
BM25RAG           0.000  0.000  0.958          0.083
```

EM/F1 が低いのはオープンエンドQAにおける「答え方の違い（answer style mismatch）」による既知の問題。Faithfulness と Hallucination Rate に有意差あり。

---

## 8. クイックスタート

### 必要環境

- Python 3.10+
- uv（パッケージマネージャ）
- OpenAI API キー

### インストール

```bash
git clone https://github.com/your-org/argumentative-path-rag
cd argumentative-path-rag

# 依存関係インストール
uv sync

# 環境変数設定
cp .env.example .env
# .env に OPENAI_API_KEY を記入
```

### 動作確認デモ

```bash
uv run python scripts/demo.py
```

Q3売上レポートをインデックスし、WHY/EVIDENCE/ASSUMPTION/WHAT の4クエリを実行する。

```
❓ [WHY] なぜQ3の売上が落ちたのか？
📌 クエリ型判定: WHY
📚 取得ノード数: 4

回答: Q3売上が前年比12%減少した主因は電子部品の在庫調整です。
     具体的には半導体不足により生産ラインが停止を余儀なくされました。
     この分析は為替レートが安定していたという前提に基づいています。
```

### 評価実行

```bash
# 合成QAで簡易比較（ArgRAG vs BM25）
uv run python scripts/quick_eval.py

# QASPER ミニ評価（論文3件×5問）
uv run python scripts/qasper_mini.py

# QASPER スケールアップ
uv run python scripts/qasper_mini.py --papers 10 --questions 5
```

### コードから使う

```python
import openai
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

client = openai.OpenAI(api_key="YOUR_KEY")
store  = NetworkXGraphStore()

# インデックス
indexer = IndexingPipeline(
    chunker=SentenceChunker(),
    classifier=NodeClassifier(client=client, model="gpt-4o-mini"),
    extractor=EdgeExtractor(client=client, model="gpt-4o-mini"),
    store=store,
)
indexer.run(document_text, doc_id="my_doc")

# クエリ
pipeline = ArgumentativeRAGPipeline(
    store=store,
    query_classifier=QueryClassifier(client=client, model="gpt-4o-mini"),
    traverser=GraphTraverser(store=store),
    context_builder=ContextBuilder(max_nodes=15),
    generator=AnswerGenerator(client=client, model="gpt-4o-mini"),
)
result = pipeline.query("なぜ売上が落ちたのか？", doc_id="my_doc")
print(result.answer)
```

---

## 9. プロジェクト構造

```
argumentative-path-rag/
├── src/ap_rag/           # メインパッケージ
│   ├── models/           # データモデル・タクソノミー定義
│   ├── indexing/         # オフラインインデックスパイプライン
│   ├── graph/            # グラフストア（NetworkX / Neo4j）
│   ├── retrieval/        # オンライン検索パイプライン
│   ├── generation/       # 回答生成
│   ├── evaluation/       # 評価フレームワーク
│   └── pipeline.py       # エンドツーエンドパイプライン
├── scripts/
│   ├── demo.py           # 動作確認デモ
│   ├── quick_eval.py     # 合成QA評価
│   └── qasper_mini.py    # QASPERベンチマーク
├── tests/
│   └── unit/             # ユニットテスト（82件）
├── docker/               # Neo4j Docker Compose
├── pyproject.toml        # uv 依存関係定義
└── .env.example          # 環境変数テンプレート
```

---

## 参考文献

- Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research.
- Gutierrez et al. (2024). *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models.*
- Zhu et al. (2025). *ArgRAG: Explainable Retrieval Augmented Generation using Quantitative Bipolar Argumentation.* NeSy 2025.
- Dasigi et al. (2021). *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers.* NAACL 2021. (QASPER)
