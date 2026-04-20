"""
インデックスパイプライン。

文書テキストを受け取り、チャンク → ノード分類 → エッジ抽出 → グラフ保存
の一連の処理を実行する。

研究計画書 §4.1「オフライン（事前処理）」フロー。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rich.progress import Progress, SpinnerColumn, TextColumn

from ap_rag.graph.store import GraphStore
from ap_rag.indexing.chunker import DocumentChunk, SentenceChunker
from ap_rag.indexing.classifier import NodeClassifier
from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.models.graph import ArgumentGraph

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """インデックス処理の結果サマリ。"""

    doc_id: str
    num_chunks: int
    num_nodes: int
    num_edges: int
    graph: ArgumentGraph


class IndexingPipeline:
    """文書のインデックス処理を実行するパイプライン。

    Args:
        chunker: テキストをチャンクに分割するクラス。
        classifier: チャンクからノードを抽出するクラス。
        extractor: ノード間のエッジを抽出するクラス。
        store: グラフを永続化するストア。
        show_progress: 進行状況をターミナルに表示するか。
    """

    def __init__(
        self,
        chunker: SentenceChunker,
        classifier: NodeClassifier,
        extractor: EdgeExtractor,
        store: GraphStore,
        show_progress: bool = True,
    ) -> None:
        self._chunker = chunker
        self._classifier = classifier
        self._extractor = extractor
        self._store = store
        self._show_progress = show_progress

    def run(self, text: str, doc_id: str) -> IndexingResult:
        """文書全体をインデックスする。

        Args:
            text: インデックス対象のテキスト。
            doc_id: 文書の一意識別子。

        Returns:
            IndexingResult — 処理結果のサマリ。
        """
        graph = ArgumentGraph(doc_id=doc_id)

        # Step 1: チャンク分割
        chunks = self._chunker.chunk(text, doc_id)
        logger.info("doc_id=%s → %d チャンク", doc_id, len(chunks))

        # Step 2 & 3: チャンクごとにノード分類 + エッジ抽出
        if self._show_progress:
            self._process_with_progress(chunks, graph)
        else:
            self._process_chunks(chunks, graph)

        # Step 4: グラフ保存
        self._store.save_graph(graph)

        result = IndexingResult(
            doc_id=doc_id,
            num_chunks=len(chunks),
            num_nodes=len(graph.nodes),
            num_edges=len(graph.edges),
            graph=graph,
        )
        logger.info(
            "インデックス完了: doc_id=%s, chunks=%d, nodes=%d, edges=%d",
            doc_id,
            result.num_chunks,
            result.num_nodes,
            result.num_edges,
        )
        return result

    # ── private ────────────────────────────────────────────────────────────

    def _process_chunks(
        self,
        chunks: list[DocumentChunk],
        graph: ArgumentGraph,
    ) -> None:
        for chunk in chunks:
            self._process_one_chunk(chunk, graph)

    def _process_with_progress(
        self,
        chunks: list[DocumentChunk],
        graph: ArgumentGraph,
    ) -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("インデックス中...", total=len(chunks))
            for chunk in chunks:
                progress.update(
                    task,
                    description=f"チャンク {chunk.chunk_idx + 1}/{len(chunks)} 処理中",
                )
                self._process_one_chunk(chunk, graph)
                progress.advance(task)

    def _process_one_chunk(
        self,
        chunk: DocumentChunk,
        graph: ArgumentGraph,
    ) -> None:
        """1チャンクのノード分類とエッジ抽出を実行してグラフに追加する。"""
        # ノード分類
        nodes = self._classifier.classify(chunk)
        for node in nodes:
            try:
                graph.add_node(node)
            except ValueError:
                # 同一テキストが重複することがあるため、重複は無視
                logger.debug("重複ノードをスキップ: %s", node.id)
                continue

        # エッジ抽出（チャンク内の有効なノードのみ対象）
        added_nodes = [n for n in nodes if n.id in graph.nodes]
        if len(added_nodes) < 2:
            return

        edges = self._extractor.extract(added_nodes)
        for edge in edges:
            try:
                graph.add_edge(edge)
            except ValueError as e:
                logger.debug("エッジ追加スキップ: %s", e)
