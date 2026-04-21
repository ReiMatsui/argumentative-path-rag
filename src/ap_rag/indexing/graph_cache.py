"""
議論グラフの永続キャッシュ。

`IndexingPipeline` が 1 文書分のグラフを作り終えた直後に JSON ファイルとして
ディスクへ書き出し、次回以降の実行で同じ文書+同じ設定なら LLM 呼び出しを
スキップしてキャッシュから復元できるようにする。

## キャッシュキー

`doc_id` だけでは不十分。グラフの中身は以下に依存するので、これらを
fingerprint に含めてキー化する:

- classifier_model: 分類器モデル (gpt-5-mini など)
- reasoning_effort: reasoning_effort (low / medium / ...)
- chunk_max_tokens: SentenceChunker の max_tokens
- use_cross_chunk: チャンク間エッジ抽出を使ったか
- prompt_version: プロンプトや抽出ロジックを変更したら手動で bump

これらのどれかが変わると新しい fingerprint になり、過去のキャッシュは
参照されなくなる (古いキャッシュファイル自体は残る — 手動で掃除すること)。

## レイアウト

    <cache_dir>/                 # 例: .cache/graphs/
      <config_hash>/             # 例: a1b2c3d4e5/
        _manifest.json           # fingerprint の中身 (デバッグ用)
        <doc_id>.json            # ArgumentGraph.model_dump_json()

`doc_id` にスラッシュ等が含まれる場合はファイル名として安全な形に変換する。

## 書き込みの原子性

`.json.tmp` に書いてから `rename` する (同一ファイルシステム上では atomic)。
こうしないと、並列実行やクラッシュで壊れた JSON が残り、次回の load で
毎回例外が出て「キャッシュ壊れてるけどなぜかヒットする」状態になる。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from ap_rag.models.graph import ArgumentGraph

logger = logging.getLogger(__name__)


# プロンプト/スキーマ/抽出ロジックの仕様を変えたら手動でここを上げる。
# 上げると古いキャッシュはヒットしなくなる。
INDEX_PROMPT_VERSION = 3


@dataclass(frozen=True)
class IndexConfigFingerprint:
    """キャッシュキーを決める identity。ここに含まれる値が同じなら再利用可能。"""

    classifier_model: str
    reasoning_effort: str
    chunk_max_tokens: int
    use_cross_chunk: bool
    prompt_version: int = INDEX_PROMPT_VERSION

    def to_hash(self) -> str:
        """10 文字の hex hash。パスとして使う分にはこれで十分衝突しない。"""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:10]


def _sanitize_doc_id(doc_id: str) -> str:
    """doc_id をファイル名として安全な形に変換する。"""
    # Windows / POSIX 両方で問題になる文字を全部 _ に
    bad = '/\\:*?"<>|'
    for c in bad:
        doc_id = doc_id.replace(c, "_")
    # 長すぎる doc_id も避ける (QASPER の paper_id は十分短いが保険)
    return doc_id[:200] if len(doc_id) > 200 else doc_id


class GraphCache:
    """ArgumentGraph の doc 単位ディスクキャッシュ。

    Args:
        cache_dir: 親ディレクトリ。存在しなければ作る。
        fingerprint: 設定 identity。同じ fingerprint のキャッシュだけを読み書きする。

    Examples:
        >>> fp = IndexConfigFingerprint(
        ...     classifier_model="gpt-5-mini",
        ...     reasoning_effort="low",
        ...     chunk_max_tokens=256,
        ...     use_cross_chunk=False,
        ... )
        >>> cache = GraphCache(".cache/graphs", fp)
        >>> if cache.has(doc_id):
        ...     graph = cache.load(doc_id)
        >>> else:
        ...     graph = indexer.run(text, doc_id).graph
        ...     cache.save(graph)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        fingerprint: IndexConfigFingerprint,
    ) -> None:
        self._fingerprint = fingerprint
        self._root = Path(cache_dir) / fingerprint.to_hash()
        self._root.mkdir(parents=True, exist_ok=True)
        self._write_manifest_once()

    # ── public ──────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    @property
    def fingerprint(self) -> IndexConfigFingerprint:
        return self._fingerprint

    def has(self, doc_id: str) -> bool:
        return self._doc_path(doc_id).exists()

    def load(self, doc_id: str) -> ArgumentGraph:
        """キャッシュ済みのグラフを読み込む。ヒットしない/壊れていれば KeyError。"""
        path = self._doc_path(doc_id)
        if not path.exists():
            raise KeyError(f"cache miss: doc_id={doc_id!r}")
        try:
            return ArgumentGraph.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as e:
            # 壊れたキャッシュは無効として例外に。呼び出し側は再生成にフォールバックする。
            logger.warning(
                "グラフキャッシュの読み込みに失敗しました (doc_id=%s, path=%s): %s",
                doc_id, path, e,
            )
            raise KeyError(f"cache corrupt: doc_id={doc_id!r}") from e

    def save(self, graph: ArgumentGraph) -> None:
        """グラフを atomic に書き出す。"""
        path = self._doc_path(graph.doc_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(graph.model_dump_json(), encoding="utf-8")
        tmp.replace(path)  # 同一 FS なら atomic
        logger.info(
            "グラフキャッシュ保存: doc_id=%s, nodes=%d, edges=%d, path=%s",
            graph.doc_id, len(graph.nodes), len(graph.edges), path,
        )

    # ── private ─────────────────────────────────────────────────────────────

    def _doc_path(self, doc_id: str) -> Path:
        return self._root / f"{_sanitize_doc_id(doc_id)}.json"

    def _write_manifest_once(self) -> None:
        """fingerprint の中身を人間可読なファイルに書いておく。

        hash だけだと後で「このディレクトリ何の設定?」が分からないので保険。
        """
        manifest = self._root / "_manifest.json"
        if manifest.exists():
            return
        try:
            manifest.write_text(
                json.dumps(asdict(self._fingerprint), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("manifest 書き込み失敗 (無視可): %s", e)
