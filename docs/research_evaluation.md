# Argumentative-Path RAG — 研究評価レポート

**作成日**: 2026-04-20
**対象**: `/Users/matsuirei/argumentative-path-rag/`（研究計画 v4 + 実装現状）
**評価軸**: (1) 新規性 vs 先行研究、(2) 手法の妥当性、(3) 評価方法の妥当性・追加提案

---

## TL;DR（忖度なしの結論）

- **新規性は "限定的だがゼロではない"**。核となる "クエリ型に応じたグラフ探索戦略の切り替え" は **PolyG (arXiv:2504.02112, Apr 2025)** に先行例があり、"因果・根拠のグラフで RAG" という発想は **CausalRAG (ACL 2025 Findings)** と強く重なる。ノード/辺の分類体系に "議論構造（CLAIM / EVIDENCE / ASSUMPTION / CAVEAT など）" を採用した点は、Teufel の Argumentative Zoning (1999/2002) と現代の Argument Mining（AQE, UniASA など）を RAG に持ち込む応用研究として **"組合せの新しさ"** はある。ただしそれだけでは弱いので、**強調すべき独自性を明確に押し出す必要**がある。
- **手法には複数の違和感**がある。特に (a) 探索戦略がハンドクラフトで、型×エッジ方向テーブルの妥当性を検証する実験がない、(b) エントリ選定を "型一致候補 → 埋め込み top-k" の2段にしているが、型分類エラーの伝播が評価されていない、(c) 7ノード × 6エッジを LLM で毎チャンク抽出するコストに対して、Longformer 系や HippoRAG 系の軽量ベースラインに勝てる見込みの議論が薄い。
- **評価方法は "最低限" だが不十分**。QASPER だけでは "議論グラフ RAG の強みが出る" と主張しにくい。QASPER 公式メトリクス（Evidence-F1）が未計測、MuSiQue / FRAMES / 2WikiMultiHopQA / LongBench v2 のような multi-hop / long-context ベンチマークが未導入、RAGAS / ARES の包括メトリクスが部分的採用に留まる。さらに **N=10 論文 × 5 問 = 50 サンプルでは統計的に差を議論できない**。

詳細を以下に展開する。

---

## 1. 新規性評価：関連研究と比較

### 1.1 本研究のコア主張（再確認）

研究計画書から読み取れるコア主張は次の3点：

1. 文書を "議論グラフ"（7ノード種別 × 6エッジ種別）として表現する。
2. クエリを5タイプ（WHY / WHAT / HOW / EVIDENCE / ASSUMPTION）に分類し、**型ごとに異なる探索戦略（入口ノード種別・探索エッジ・方向・深さ）**を用いる。
3. これにより、意味ベクトル類似では取れない "根拠チェーン" を回収し、WHY/EVIDENCE 系質問に強い RAG を作る。

### 1.2 直接的に競合する先行研究

| # | 研究 | 年 | 本研究との重なり | 差分として残せるもの |
|---|---|---|---|---|
| A | **PolyG — Adaptive Graph Traversal for Diverse GraphRAG Questions** ([arXiv:2504.02112](https://arxiv.org/abs/2504.02112)) | 2025/04 | **クエリ型タクソノミに応じた適応的グラフ探索** という発想そのもの。4クラスの質問分類→LLM が Cypher を動的生成。レイテンシ4x改善・トークン95%削減・75% win rate を報告。 | PolyG は KGQA（エンティティ中心 KG, Freebase 等）。本研究は **文書側が "議論グラフ"** で、ノード種別が議論構造。クエリ型も "WHY/EVIDENCE/ASSUMPTION" のように議論機能に寄せている。→ "議論ドメインでの PolyG" と言い換えられるなら新規性は維持可能。 |
| B | **CausalRAG — Integrating Causal Graphs into RAG** ([ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1165/)) | 2025/03 | **因果三つ組を抽出 → 因果グラフ構築 → 因果パス検索 → 因果制約付き生成**。WHY 系の質問を意識。グラフ RAG の GraphRAG 系ベースラインに勝つと主張。 | 本研究は "因果" に閉じず、ASSUMES / CONTRADICTS / ILLUSTRATES まで含む広い議論関係を扱う。加えて **ASSUMPTION ノードの明示** は CausalRAG にない。→ "因果より広い議論関係を RAG に持ち込む" で差別化可能。 |
| C | **PathRAG — Pruning Graph-Based RAG with Relational Paths** ([arXiv:2502.14902](https://arxiv.org/pdf/2502.14902)) | 2025/02 | ノードではなく **"関係パス"** を検索単位にする発想。flow-based reliability pruning。 | 本研究はパス "そのもの" を検索する設計ではなく、"型に応じた BFS 展開" で複数パスを暗黙に回収する。→ ここでの差別化は構造上あるが、"なぜ BFS でパスよりも良いのか" は実験で示す必要。 |
| D | **Paths-over-Graph (PoG)** ([WWW 2025](https://dl.acm.org/doi/10.1145/3696410.3714892)) | 2025 | LLM 推論にパスを与えるという発想。平均 +18.9% 改善を報告。 | 同上：エンティティ中心 KG ベースなので議論グラフではない。 |
| E | **HippoRAG / LightRAG / Microsoft GraphRAG** | 2024–2025 | Graph RAG のメインライン。階層コミュニティ or 双層検索 or 記憶想起アナロジー。 | いずれも **エンティティ中心 KG**。議論構造は扱わない。→ 本研究の差別化軸。 |
| F | **Argumentative Zoning (Teufel 1999, 2002)** ([AZ corpus](https://www.cl.cam.ac.uk/~sht25/az.html)) | 1999–2009 | 学術文章の文を **Aim / Background / Own / Contrast / Basis / Other / Textual** に分類。15クラス拡張版もある。 | "既存の AZ スキーマ" ではなく、本研究は **7ノード×6エッジの議論タクソノミ** を独自設計。AZ を RAG 検索に接続した前例は希少（サーベイ範囲内で直接例なし）。 |
| G | **Argument Mining / UniASA / AQE** ([UniASA MIT 2025](https://direct.mit.edu/coli/article/51/3/739/127893/UniASA-A-Unified-Generative-Framework-for-Argument), [LLM in AM Survey arXiv:2506.16383](https://arxiv.org/html/2506.16383v4)) | 2020–2025 | 議論要素（claim / premise）と関係（support / attack）を抽出。 | "抽出した議論構造を検索に使う" ところまでやる研究は少ない。PaperTrail（scholarly QA の claim-evidence マッチング）が近いが、"グラフ探索 + 型適応戦略" は統合していない。 |

### 1.3 新規性の総合評価（忖度なし）

- **完全な新規性**: ほぼない。"議論タクソノミ × クエリ型適応 × グラフ探索 × RAG" の組合せは存在しないが、**各要素は既に先行研究がある**。
- **組合せの新規性**: **中程度**。特に「PolyG（KGQA ドメイン）を議論ドメインに持ち込み、ASSUMPTION や CAVEAT のような "暗黙の前提・留保" まで明示ノード化する」という設計は新しい。
- **主張として強く押し出せる軸**:
  1. **"暗黙の前提 (ASSUMPTION) を明示ノード化" する RAG** → WHY / ASSUMPTION 型質問で効く、という仮説の実証ができれば PolyG / CausalRAG との差別化になる。
  2. **"議論関係（SUPPORTS / CONTRADICTS / DERIVES / ILLUSTRATES）で階層的に根拠を辿る"** → 通常の entity-KG では表現されない関係ラベル。
  3. **"クエリ型 × エッジ型 × 方向 × 深さ" を組み合わせた探索戦略テーブル** → PolyG は LLM に Cypher を動的生成させるが、本研究は **ルールベースで決定的かつ解釈可能**。

- **弱点として認識すべき点**:
  - PolyG / CausalRAG が直接の競合である以上、「これらをベースラインとして勝つ」ことを示さないと査読で落ちる可能性が高い。現状の baselines（BM25, DenseRAG, VanillaRAG）だけでは不十分。
  - "議論ドメイン" と言っても QASPER は NLP 論文の事実抽出系 QA が多く、WHY / ASSUMPTION が支配的ではない。→ ベンチマーク選定の見直し（後述）が必要。

**推奨される論文上のポジショニング**：
> "GraphRAG・PolyG 系が扱ってきた entity-centric KG に対し、本研究は文書の argumentative structure (claim/evidence/assumption/...) を一級市民として扱う。PolyG のクエリ型適応を argumentative 関係（SUPPORTS/CONTRADICTS/DERIVES/ASSUMES）に拡張し、特に暗黙の前提 (ASSUMPTION) を明示ノード化することで、WHY / ASSUMPTION 型質問における根拠回収を改善する。"

---

## 2. 手法の妥当性レビュー

### 2.1 良い点（維持すべき）

- **7ノード × 6エッジの taxonomy** は小さく保たれており、LLM 抽出の負荷が現実的。
- **Structured Outputs** による JSON 強制で、抽出安定性が確保されている。
- **並列インデックス（`max_workers=4`）** はコスト面で妥当。
- **E5-Mistral-7B-Instruct を入口ノード絞り込みに使う** のは 2024–2025 の標準的ベストプラクティスに合致。
- **DenseRAG / BM25RAG / VanillaRAG ベースラインの内製化** は比較の内的一貫性を保つ上で重要。
- **answer-consistency（ペアワイズ F1 平均）の導入** は評価の再現性に対する意識の高さを示しており、最近の議論（["Improving Consistency in RAG with Group Similarity Rewards" arXiv:2510.04392](https://arxiv.org/html/2510.04392v1)）とも整合。
- **cross-chunk edge extraction（Step 3.5）** は、long-document のチャンク境界による "根拠の断絶" 問題に対する妥当な処方。

### 2.2 方法論的な違和感（設計レベル）

#### ① 探索戦略テーブルが "手で書いた正解表" になっている

`TRAVERSAL_STRATEGIES` は `QueryType → (entry_node_types, edge_types, directions, depth)` を固定マッピングしている。ここにはいくつか懸念がある：

- **"WHY なら EVIDENCE→CLAIM を逆方向に辿る" が本当に最適かは未検証**。QASPER で 5–10 個の実際の WHY 質問を眺めて、手設計した戦略が妥当かを質的に確認する "chase analysis" がない。
- **PolyG は LLM に動的生成させる**ことでこの問題を回避している。本研究の "ルールベース" を積極的に擁護するなら、**"解釈可能性・決定性・コスト" で LLM-dynamic 方式に勝つ根拠**を論文で主張する必要がある。
- 最悪ケースとして、**戦略表の1行を変えるだけで結果が大きく動く** 可能性があり、そうだとすれば主張は "taxonomy の勝利" ではなく "ハイパラチューニングの勝利" になる。→ **戦略表のセンシティビティ分析 (ablation)** を必ず追加すべき。

**推奨**: 戦略表の各行（各 QueryType）について、entry_node_types を変えた、edge_types を外した、depth を変えた場合の性能差を ablation で示す。

#### ② ノード分類のエラー伝播が評価されていない

- `NodeClassifier` が CLAIM と EVIDENCE を取り違えると、探索戦略はそもそも誤った "型" のノードを起点に展開する。
- 現状 `qasper_mini.py` と `quick_eval.py` は **end-to-end 精度しか見ていない**。→ 分類器単体の精度（LLM as judge or 人手小サンプル）を測って、分類器の誤りが retrieval/answer 精度にどれだけ波及するかを切り分けるべき。
- これは単なる "ablation" ではなく、主張の妥当性に直結する。"taxonomy が効いている" と言うには、"分類が正しい限り戦略が効く" を示す必要がある。

#### ③ Entry selection の2段構え（型フィルタ → 埋め込み top-k）は設計上の綻びがある

`pipeline.py: _select_entry_nodes` の現在の挙動：

```
候補 = 型一致ノード全件 (例: CLAIM ノード 150 個)
→ EmbeddingNodeSelector で top-10 に絞る
```

懸念：

- **型分類に偽陰性があると、候補集合から該当ノードが落ちる**（埋め込みで救えない）。
- 逆に、**型分類は正しくても質問と表層が離れていると埋め込みも外す**（とくに科学論文で "we hypothesize that..." が CLAIM として抽出されたが、質問は言い換え）。
- **型適応戦略と埋め込み選定が独立** に動いているので、「型 A が選ばれているのに、入口では型 A のノードが候補にない」状況が起きる（空回り）。→ フォールバック（別型で検索）が必要か検討。

**推奨**: `BenchmarkDebug` を追加して、QASPER 評価中に "型別入口候補数 / 型別選定数 / 空候補時のフォールバック回数" をログに出し、失敗サンプルの原因分布を分析できるようにする。

#### ④ クロスチャンクエッジの閾値 `min_similarity=0.45` はマジックナンバー

- 0.45 で取れる候補数が論文ごとに大きくブレる可能性が高い。
- 現状は embedding top-5 をバッチ verify する設計だが、**verify step の precision/recall が計測されていない**（LLM が "関係あり" と言ったものがどれだけ正しいか）。
- **この追加エッジが結果を改善しているか** は実測が必要（`--cross-chunk` on/off の ablation）。

**推奨**: `--cross-chunk` の有無で同じ QASPER サブセットを走らせ、Answer-F1 / Evidence-F1 / retrieval recall@K の差分を報告する。

#### ⑤ "QueryType" の粒度が議論ドメインに対して粗い可能性

- 現状は 5 クラス（WHY / WHAT / HOW / EVIDENCE / ASSUMPTION）。
- QASPER の質問分布を見ると "method comparison"（どの手法が優れていたか）、"quantitative"（数値）、"yes/no" が多く、この 5 クラスのどれにもうまくハマらない質問が存在する。
- **PolyG は 4 クラス（specified attr / unspecified attr / relation / existence）で、これが KGQA では支配的クラス**。本研究の 5 クラスは議論系クラスに寄っており、QASPER の実分布と乖離している可能性。

**推奨**: QASPER の validation の質問 100 件を 5 クラスに人手分類し、分布と "unclassifiable" 率を報告する。5 クラスが QASPER の主要クラスをカバーしていないなら、QASPER は主要ベンチマークとしては不適（MuSiQue や FRAMES のほうが合う可能性）。

#### ⑥ コスト vs 性能の議論が手薄

- 7 ノード × 6 エッジを LLM で毎チャンク抽出するので、**インデックス時のコスト**が大きい。
- Microsoft GraphRAG は大規模で $33K のインデックスコストで話題になり、LightRAG や HippoRAG はこれを下げる方向で差別化している。本研究は "taxonomy による表現力" を売りにするなら、"LightRAG 比でどれだけコスト増でどれだけ精度向上" を数値で示さないと、査読者が最初に突く。
- 現状のベースラインに **HippoRAG / LightRAG / GraphRAG 系が含まれていない** のは重大な穴。

**推奨**: 少なくとも `nano-graphrag`（MS GraphRAG の軽量実装）または `LightRAG` を baseline に追加し、"コスト vs 精度" の Pareto 点を論じる。

### 2.3 実装レベルで確認すべき細部

- `infer_query_type` のキーワードルール（`qasper.py:26-31`）は fallback 用として残っているが、LLM 分類器がある場合は差分（LLM vs rule）をログに出して分類一貫性を確認した方が良い。
- `EmbeddingNodeSelector._format_passage` は `{node_type.value}: {node.text}` を使っており、E5-Mistral のパッセージ側にプレフィックスを付けている。E5-Mistral 公式は passage にプレフィックス不要としており、**ここは A/B を比較して判断する価値がある**（prefix が悪影響の可能性も）。
- `TRAVERSAL_STRATEGIES[QueryType.WHAT]` の depth 0→2 逸脱（実装と plan が違う）が研究計画 v5 patch で既に記録されている通り。これは論文側で "explicitly deviating from prior plan because …" と説明する必要がある。

---

## 3. 評価方法の妥当性と追加提案

### 3.1 現状の評価構成（確認）

- **ベンチマーク**: QASPER（validation split, 少数論文 + 少数質問）。
- **メトリクス**: Exact Match、F1（トークン重複）、LLM-judge（faithfulness / hallucination / answer correctness）、answer consistency（ペアワイズ F1 平均）。
- **ベースライン**: VanillaRAG（未適応 LLM 直接回答）、BM25RAG、DenseRAG、本手法（AP-RAG）。

### 3.2 最大の弱点（優先度順）

#### ① QASPER の **Evidence-F1** が計測されていない（最優先）

QASPER 公式評価には **Answer-F1 と Evidence-F1 の両方** がある。Evidence-F1 は "支持根拠として選ばれたパラグラフが正解と一致するか" を測る。本研究のコア仮説は "議論グラフ RAG は根拠を正しく辿れる" なので、**Evidence-F1 こそ本研究の主張を最も直接検証するメトリクス**。現状これが未計測なのは大きな機会損失。

**推奨対応**:
- QASPER データ構造に含まれる `evidence` フィールド（段落単位）を `EvaluationSample.evidence_paragraphs` として持たせる。
- `retrieved_contexts` と evidence のパラグラフ単位 overlap で Evidence-F1 を計算。
- ベースラインに対して "Answer-F1 は同等でも Evidence-F1 が上" なら強い主張になる。

#### ② 統計的パワー（サンプル数）が不足

- 既定の `num_papers=10, num_questions=5` → **最大 50 サンプル**。
- RAG 系の改善幅は Answer-F1 で ±2〜5 pt が一般的。50 サンプルでは標準誤差 ~5pt 以上で差が判定不能。
- **最低でも 500 サンプル**（論文 50 × 質問 10）が欲しい。推奨 1000〜2000。
- コストが問題なら、LLM-judge のみ抽出サンプルで、Answer-F1 / EM は大規模サンプルで測る "二層評価" が妥当。

#### ③ LLM-as-Judge のバイアス対策

[Confident AI / arXiv:2412.05579](https://arxiv.org/html/2412.05579v2) などで報告されているように、LLM judge には：

- **冗長性バイアス**（長い回答を高く評価, +~15%）
- **自己選好バイアス**（自分の出力を高く評価）
- **位置バイアス**（pairwise で ~40% の非一貫性）
- **追従バイアス**（assertive な主張を過信）

現在の `LLMJudge` はおそらく単一 GPT-4 系モデルでの評価と思われる。

**推奨対応**:
- ペアワイズ比較の際は **順序を入れ替えて二重評価**し一致率を見る。
- **複数判定器**（GPT-4o + Claude + Gemini）の多数決。
- または **FaithJudge**（人手アノテーションを参照した faithfulness 判定器）方式を一部採用。

### 3.3 追加で入れるべきベンチマーク（優先度順）

| 優先度 | ベンチマーク | なぜ必要か | リンク |
|---|---|---|---|
| ★★★ | **MuSiQue** | 真の multi-hop を保証（single-hop で解けない設計）。"議論チェーンを辿る" の主張検証に直結。単一パラグラフ baseline で ~32 F1 という難度は本研究の差が出やすい。 | [EmergentMind: MuSiQue](https://www.emergentmind.com/topics/musique-condition) |
| ★★★ | **FRAMES** (Google, 824 Q, 2–15 docs/Q) | factuality + retrieval + reasoning を統合評価。single-step で ~40%、multi-step で ~66%。本研究の "根拠グラフ" 設計が効くはず。 | [arXiv:2409.12941](https://arxiv.org/abs/2409.12941) |
| ★★ | **2WikiMultiHopQA** | 因果・構成的推論を強調。CausalRAG 系と直接比較しやすい。 | — |
| ★★ | **LongBench v2** | 長文脈の多タスク評価。QASPER 系 QA も含む。`allenai/qasper` だけだとドメインが NLP 論文に偏る。 | [ACL 2025](https://aclanthology.org/2025.acl-long.183.pdf) |
| ★ | **GraphRAG-Bench (ICLR'26)** | "When to use Graphs in RAG" の質問難度階層付きベンチ。Graph RAG 系の公平比較に使える。 | [GitHub](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) |
| ★ | **HotpotQA** | multi-hop の定番。ただし MuSiQue より古く shortcut で解かれる批判がある。参考値として。 | — |
| △ | **MMLongBench-Doc** | 長文 PDF（49ページ/doc）。ただし multimodal 前提で本研究の純テキスト設計とは追加コストが高い。優先度低。 | [arXiv:2407.01523](https://arxiv.org/abs/2407.01523) |

### 3.4 追加で採用すべきメトリクス

| メトリクス | 由来 | なぜ必要か |
|---|---|---|
| **Evidence-F1** | QASPER 公式 | §3.2 ① で詳述。本研究の主張に直結。**必須**。 |
| **Context Precision / Context Recall** | [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) | 取得コンテキストの質を分離して測る。Answer-F1 が低い原因が "retrieval" か "generation" かを切り分けるのに必要。 |
| **Context Relevance** | ARES | 同上、判定器ベース版。 |
| **Response Groundedness / Faithfulness** | RAGAS / FaithJudge | 現状の LLM-judge の置き換え候補。より標準化された定義。 |
| **Retrieval Recall@K (根拠カバレッジ)** | 一般 IR | 正解 evidence の K 件以内ヒット率。Evidence-F1 の前段指標。 |
| **Latency / Token cost / Index cost** | HippoRAG / LightRAG 論文の慣行 | §2.2 ⑥ で述べたコスト議論に必須。 |
| **Per-query-type breakdown (WHY / EVIDENCE / …)** | 本研究独自 | 既に実装済みだが、サンプル数増やして有意性を出す必要。 |

### 3.5 必須の ablation（論文として最低限）

1. **Taxonomy ablation**: ノード種別を 7 → 3（CLAIM / EVIDENCE / その他）に減らしたとき、エッジ種別を 6 → 2（SUPPORTS / その他）に減らしたときの Answer-F1 / Evidence-F1 変化。
2. **Strategy table ablation**: 各 QueryType について探索戦略を "全ノード対象 / 全エッジ種別 / depth=固定" の一律設定に変えたときの性能。
3. **Entry selection ablation**: BM25 / Dense / 型フィルタ+Dense の3条件比較。
4. **Cross-chunk ablation**: `--cross-chunk` on/off の比較。
5. **Query classifier ablation**: LLM 分類 vs キーワードルールで end-to-end 性能がどれだけ変わるか（これは既に `qasper_mini.py` で切替可能なので実行のみ）。
6. **Baseline comparison**: **PolyG, CausalRAG, LightRAG または GraphRAG** のいずれかを再現 or 公式実装で動かして比較。これなしでは論文の novelty 主張が通らない。

---

## 4. 総合評価（Executive Summary）

| 観点 | 評価 | コメント |
|---|---|---|
| 新規性 | ★★☆☆☆ | PolyG (2025/04) と CausalRAG (2025/03) に先行され、Graph×adaptive×RAG 単独では通らない。"議論 taxonomy を持ち込み ASSUMPTION を明示化" で差別化。 |
| 手法の妥当性 | ★★★☆☆ | 設計は筋が通っているが、戦略テーブルの正当化と分類器エラー伝播の評価が欠けている。 |
| 実装の堅牢性 | ★★★★☆ | 並列インデックス、structured output、エンコーダ共有などは適切。answer consistency 評価は新しい実装として好ましい。 |
| 評価の十分性 | ★★☆☆☆ | QASPER のみ + Evidence-F1 未計測 + サンプル数 50 では論文として弱い。最低でも MuSiQue / FRAMES の追加が必要。 |
| 結果の再現性・比較可能性 | ★★★☆☆ | ベースライン内製は良いが、PolyG / CausalRAG / LightRAG などとの外部比較がないと第三者視点で勝ち負けが不明。 |

### 次に取るべきアクション（優先度順）

1. **Evidence-F1 の実装**（QASPER の `evidence_paragraphs` を `EvaluationSample` に入れて計測）。— 最大のレバレッジ。
2. **MuSiQue と FRAMES へのベンチマーク拡張**（runner 追加）。— 新規性主張を支える。
3. **PolyG or CausalRAG のどちらかと直接比較**（公式実装を動かす or ネイティブに近い再実装）。
4. **Strategy table の ablation**（6種類以上の条件で 500 サンプル）。
5. **サンプル数を 500–2000 に拡大**、LLM-judge は抽出サンプルで評価。
6. **戦略テーブルの設計根拠**を論文 §3 に明記（decision log として残す）。

---

## Sources

- [PolyG: Adaptive Graph Traversal for Diverse GraphRAG Questions](https://arxiv.org/abs/2504.02112)
- [CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.1165/)
- [PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths](https://arxiv.org/pdf/2502.14902)
- [Paths-over-Graph: Knowledge Graph Empowered LLM Reasoning (WWW 2025)](https://dl.acm.org/doi/10.1145/3696410.3714892)
- [Retrieval-Augmented Generation with Graphs (GraphRAG survey)](https://arxiv.org/html/2501.00309v2)
- [GraphRAG-Bench: When to use Graphs in RAG (ICLR'26)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [UniASA: A Unified Generative Framework for Argument Structure Analysis (Comp Ling 2025)](https://direct.mit.edu/coli/article/51/3/739/127893/UniASA-A-Unified-Generative-Framework-for-Argument)
- [Large Language Models in Argument Mining: A Survey](https://arxiv.org/html/2506.16383v4)
- [Teufel & Moens — Argumentative Zoning corpus & scheme](https://www.cl.cam.ac.uk/~sht25/az.html)
- [QASPER: A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers](https://arxiv.org/abs/2105.03011)
- [FRAMES: Fact, Fetch, and Reason (Google DeepMind)](https://arxiv.org/abs/2409.12941)
- [MuSiQue condition summary (EmergentMind)](https://www.emergentmind.com/topics/musique-condition)
- [MMLongBench-Doc: Long-context Document Understanding](https://arxiv.org/abs/2407.01523)
- [LongBench v2 (ACL 2025)](https://aclanthology.org/2025.acl-long.183.pdf)
- [SciQA Scientific Question Answering Benchmark (Nature Scientific Reports)](https://www.nature.com/articles/s41598-023-33607-z)
- [RAGAS metrics documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [Improving Consistency in RAG with Group Similarity Rewards (arXiv 2510.04392)](https://arxiv.org/html/2510.04392v1)
- [LLMs-as-Judges: A Comprehensive Survey (arXiv 2412.05579)](https://arxiv.org/html/2412.05579v2)
- [Benchmarking LLM Faithfulness in RAG (arXiv 2505.04847)](https://arxiv.org/html/2505.04847v2)
- [Graph RAG in 2026: Practitioner's Guide (Graph Praxis)](https://medium.com/graph-praxis/graph-rag-in-2026-a-practitioners-guide-to-what-actually-works-dca4962e7517)
