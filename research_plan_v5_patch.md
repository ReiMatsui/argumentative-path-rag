# 研究計画書 v5 パッチ — エッジ方向規約の修正

## 目的

実装（`src/ap_rag/models/taxonomy.py`）と研究計画書 v5 の図・本文の間に、
エッジの **向き** の規約にズレがある。実装側の向きの方が意味的に自然（矢印
がそのまま "この情報を根拠にこの情報が成立する" という支持関係を示す）ため、
**研究計画書の方を実装に合わせて修正する** ためのパッチを以下にまとめる。

Google Doc 上で該当箇所を探し、このパッチどおりに置換すれば整合が取れる。

---

## 実装側の規約（これが正）

| エッジ型 | 向き | 意味 |
|:---|:---|:---|
| `SUPPORTS`    | **EVIDENCE → CLAIM**         | 根拠が主張を支持する（矢印の先が支持される側） |
| `DERIVES`     | **EVIDENCE → CONCLUSION**    | 根拠から結論が論理的に導出される |
| `ASSUMES`     | **CLAIM → ASSUMPTION**       | 主張が前提を要請する（矢印の先が前提） |
| `ILLUSTRATES` | **visual/example → CLAIM**   | 具体例が主張を例示する |
| `CONTRASTS`   | **CONTRAST → CLAIM**         | 対比ノードが主張と対比される |
| `CONTRADICTS` | **CONTRAST → CLAIM**         | 対比・反例が主張に反する |

**要点**: 「矢印の先にあるノードは、矢印の元のノードによって支えられている／
前提とされている側」というルールで全エッジが一貫している。

---

## §2.1 グラフ図の修正（**図の矢印向きの置換**）

### 修正前（v5 の現行図）

```
CLAIM ──SUPPORTS──▶ EVIDENCE
CLAIM ──DERIVES───▶ CONCLUSION
ASSUMPTION ──ASSUMES──▶ CLAIM
```

### 修正後

```
EVIDENCE   ──SUPPORTS──▶ CLAIM
EVIDENCE   ──DERIVES───▶ CONCLUSION
CLAIM      ──ASSUMES───▶ ASSUMPTION
CONTRAST   ──CONTRASTS─▶ CLAIM
CONTRAST   ──CONTRADICTS▶ CLAIM
```

説明文（図の下）も以下のように書き換える:

> エッジは「支える側 → 支えられる側」「前提を要求する側 → 前提」を向く。
> たとえば `SUPPORTS` は EVIDENCE（支える側）から CLAIM（支えられる側）に
> 向かう。探索時は CLAIM を入口としたとき、`SUPPORTS` を **逆向き (incoming)** に
> たどることで支持する EVIDENCE を取得する。

---

## §2.3 エッジ規約テーブルの修正

現行テーブルの向き欄を以下に置換する:

| エッジ型       | 矢印（src → tgt）               | セマンティクス |
|:---|:---|:---|
| `SUPPORTS`    | EVIDENCE → CLAIM              | src が tgt の主張を支持する |
| `CONTRADICTS` | CONTRAST → CLAIM              | src が tgt の主張に反する |
| `DERIVES`     | EVIDENCE → CONCLUSION         | src から tgt が論理的に導出される |
| `ASSUMES`     | CLAIM → ASSUMPTION            | tgt は src の成立を前提とする（tgt を暗黙に要請する） |
| `ILLUSTRATES` | visual/example → CLAIM        | src が tgt を具体例・図表で示す |
| `CONTRASTS`   | CONTRAST → CLAIM              | src が tgt と対比関係にある |

**補足文**（テーブルの直後に追加推奨）:

> すべてのエッジは **「根拠的・前提的なノード → 支持／前提される側のノード」**
> という一貫した向きで定義される。本文以降で「CLAIM から SUPPORTS を辿る」と
> 書くとき、これは SUPPORTS エッジを **逆向き (incoming)** に辿ることを意味する。

---

## §4.2 クエリ型ごとの探索戦略テーブルへの加筆

各クエリ型の「follow_edges」列に **方向 (incoming/outgoing)** の列を追加する。
実装値は以下のとおり:

| クエリ型     | 入口ノード                          | 辿るエッジ (方向)                                                    | 除外        | 最大深度 |
|:---|:---|:---|:---|:---|
| WHY          | CLAIM, CONCLUSION                 | SUPPORTS (in), DERIVES (in), ASSUMES (out)                          | CONTRAST    | 3        |
| WHAT         | CLAIM, DEFINITION, CONCLUSION     | SUPPORTS (in), DERIVES (in)                                         | —           | 2        |
| HOW          | CLAIM, CONCLUSION                 | SUPPORTS (in), DERIVES (in)                                         | —           | 3        |
| EVIDENCE     | CLAIM                             | SUPPORTS (in)                                                        | CONTRAST, ASSUMPTION | 2 |
| ASSUMPTION   | CLAIM, CONCLUSION                 | ASSUMES (out)                                                        | —           | 2        |

**in = incoming（エッジの向きを逆にたどる）** / **out = outgoing（エッジの向きどおりにたどる）**。

### 読み方の例

- WHY で CLAIM を入口にしたとき、`SUPPORTS (in)` は
  「CLAIM に向かって SUPPORTS で入ってくる EVIDENCE を拾う」ことを意味する。
- ASSUMPTION で CLAIM を入口にしたとき、`ASSUMES (out)` は
  「CLAIM から ASSUMES で出ていく ASSUMPTION を拾う」ことを意味する。

---

## §4.2 補足（v5 からの逸脱に関する実装メモ）

実装では計画書 v5 の戦略から以下 2 点だけ意図的に逸脱している。研究計画書に
**「実装との差分」** という小見出しを新設し、以下を明記しておくと後で自分が
混乱しない:

- **WHAT 型の depth=0 → depth=2 への変更**
  計画書原案では WHAT は入口 CLAIM/DEFINITION のみで深度 0 としていたが、
  QASPER の予備実験で WHAT が全体の約 86% を占め、かつ BM25 に大幅に劣る
  結果となった。原因は EVIDENCE が完全に除外されていたため。現実装では
  `SUPPORTS/DERIVES (incoming) depth=2` で EVIDENCE も取得する。
  CONCLUSION も入口に追加。

- **HOW 型の ILLUSTRATES → SUPPORTS/DERIVES への置き換え**
  視覚ノード前提の `ILLUSTRATES` はテキスト論文にほぼ存在しなかったため、
  `SUPPORTS/DERIVES (incoming)` に置き換えて手順を支持する EVIDENCE を
  取得する形に変更した。

どちらも「元の計画書の意図（型に応じた情報選別）」は保ったまま、辺の型名を
実データに合わせて差し替えただけであり、研究の主張には影響しない。

---

## 該当する図・本文の検索キーワード

Google Doc の検索 (Cmd/Ctrl+F) で該当箇所を見つけるためのキー:

- `CLAIM ──SUPPORTS──▶ EVIDENCE` / `SUPPORTS` を含む矢印図
- `ASSUMPTION ──ASSUMES──▶ CLAIM` / `ASSUMES` を含む矢印図
- 「src が tgt を支持する」などのエッジ定義表
- 「クエリ型ごとの探索戦略」テーブル

---

## 影響を受けない箇所

以下の箇所は実装と一致しているので触らなくてよい:

- ノード種別（CLAIM, EVIDENCE, ASSUMPTION, CONCLUSION, CAVEAT, CONTRAST,
  DEFINITION の 7 種）
- クエリ型（WHY, WHAT, HOW, EVIDENCE, ASSUMPTION の 5 種）
- 評価指標の定義（EM/F1, Faithfulness, Hallucination率, 一貫性, 引用精度）
- オフライン／オンラインフローの全体像
