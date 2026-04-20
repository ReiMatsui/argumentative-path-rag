"""
インデックスパイプライン。

文書テキストを受け取り、チャンク → ノード分類 → エッジ抽出 → グラフ保存
の一連の処理を実行する。

研究計画書 §4.1「オフライン（事前処理）」フロー。

並列化:
    NodeClassifier / EdgeExtractor の呼び出しは I/O バウンド（OpenAI API 待ち）
    なので ThreadPoolExecutor で並列実行できる。グラフへの書き込みだけを
    threading.Lock で直列化することで安全性を確保する。
    デフォルト max_workers=4 は一般的な OpenAI Tier でレートリミットに
    かからない安全な並列数。
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ap_rag.graph.store import GraphStore
from ap_rag.indexing.chunker import DocumentChunk, SentenceChunker
from ap_rag.indexing.classifier import NodeClassifier
from ap_rag.indexing.cross_chunk import CrossChunkEdgeExtractor
from ap_rag.indexing.extractor import EdgeExtractor
from ap_rag.models.graph import ArgumentEdge, ArgumentGraph, ArgumentNode

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """インデックス処理の結果サマリ。"""

    doc_id: str
    num_chunks: int
    num_nodes: int
    num_edges: int
    graph: ArgumentGraph


# チャンク単位の処理結果（スレッド → メインスレッドへの受け渡し用）
@dataclass
class _ChunkResult:
    chunk_idx: int
    nodes: list[ArgumentNode]
    edges: list[ArgumentEdge]


class IndexingPipeline:
    """文書のインデックス処理を実行するパイプライン。

    Args:
        chunker: テキストをチャンクに分割するクラス。
        classifier: チャンクからノードを抽出するクラス。
        extractor: ノード間のエッジを抽出するクラス。
        store: グラフを永続化するストア。
        show_progress: 進行状況をターミナルに表示するか。
        max_workers: 並列 API 呼び出し数。
                     OpenAI の一般的な Tier でレートリミットに
                     かからない目安として 4 をデフォルトとする。
                     直列処理にしたい場合は 1 を指定。
    """

    def __init__(
        self,
        chunker: SentenceChunker,
        classifier: NodeClassifier,
        extractor: EdgeExtractor,
        store: GraphStore,
        show_progress: bool = True,
        max_workers: int = 4,
        cross_chunk_extractor: CrossChunkEdgeExtractor | None = None,
    ) -> None:
        self._chunker = chunker
        self._classifier = classifier
        self._extractor = extractor
        self._store = store
        self._show_progress = show_progress
        self._max_workers = max_workers
        self._cross_chunk_extractor = cross_chunk_extractor

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
        logger.info("doc_id=%s → %d チャンク (max_workers=%d)", doc_id, len(chunks), self._max_workers)

        # Step 2 & 3: チャンクごとにノード分類 + エッジ抽出（並列）
        if self._show_progress:
            self._process_parallel_with_progress(chunks, graph)
        else:
            self._process_parallel(chunks, graph)

        # Step 3.5: チャンクをまたぐエッジの補完（任意）
        if self._cross_chunk_extractor is not None:
            self._run_cross_chunk_step(graph)

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

    def _run_cross_chunk_step(self, graph: ArgumentGraph) -> None:
        """チャンク間エッジ抽出器を走らせて新規辺を `graph` に追加する。"""
        assert self._cross_chunk_extractor is not None
        all_nodes = list(graph.nodes.values())
        existing_edges = list(graph.edges.values())

        before = len(existing_edges)
        try:
            new_edges = self._cross_chunk_extractor.extract(all_nodes, existing_edges)
        except Exception as e:
            logger.warning("CrossChunkEdgeExtractor 失敗: %s", e)
            return

        for edge in new_edges:
            if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
                continue
            try:
                graph.add_edge(edge)
            except ValueError as e:
                logger.debug("チャンク間エッジ追加スキップ: %s", e)

        logger.info(
            "チャンク間エッジ補完: +%d 辺 (合計 %d → %d)",
            len(new_edges),
            before,
            len(graph.edges),
        )

    def _call_llm_for_chunk(self, chunk: DocumentChunk) -> _ChunkResult:
        """1チャンクの LLM 処理（classify + extract）をスレッド内で実行する。

        グラフへの書き込みは行わず、結果を返すだけにすることで
        スレッドセーフ性を確保する。
        """
        nodes = self._classifier.classify(chunk)

        edges: list[ArgumentEdge] = []
        if len(nodes) >= 2:
            edges = self._extractor.extract(nodes)

        return _ChunkResult(chunk_idx=chunk.chunk_idx, nodes=nodes, edges=edges)

    def _write_chunk_result(
        self,
        result: _ChunkResult,
        graph: ArgumentGraph,
        lock: threading.Lock,
    ) -> None:
        """チャンク処理結果をグラフに書き込む（ロックで直列化）。"""
        with lock:
            for node in result.nodes:
                try:
                    graph.add_node(node)
                except ValueError:
                    logger.debug("重複ノードをスキップ: %s", node.id)

            for edge in result.edges:
                # 両端ノードがグラフに存在する場合のみ追加
                if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
                    try:
                        graph.add_edge(edge)
                    except ValueError as e:
                        logger.debug("エッジ追加スキップ: %s", e)

    def _process_parallel(
        self,
        chunks: list[DocumentChunk],
        graph: ArgumentGraph,
    ) -> None:
        """進行状況表示なしの並列処理。"""
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_chunk: dict[Future[_ChunkResult], DocumentChunk] = {
                executor.submit(self._call_llm_for_chunk, chunk): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                except Exception as e:
                    logger.warning("チャンク %d の処理でエラー: %s", chunk.chunk_idx, e)
                    continue
                self._write_chunk_result(chunk_result, graph, lock)

    def _process_parallel_with_progress(
        self,
        chunks: list[DocumentChunk],
        graph: ArgumentGraph,
    ) -> None:
        """進行状況バー付きの並列処理。"""
        lock = threading.Lock()
        total = len(chunks)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"インデックス中… (並列数={self._max_workers})",
                total=total,
            )

            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_chunk: dict[Future[_ChunkResult], DocumentChunk] = {
                    executor.submit(self._call_llm_for_chunk, chunk): chunk
                    for chunk in chunks
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_result = future.result()
                    except Exception as e:
                        logger.warning("チャンク %d の処理でエラー: %s", chunk.chunk_idx, e)
                        progress.advance(task)
                        continue

                    self._write_chunk_result(chunk_result, graph, lock)
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"インデックス中… (並列数={self._max_workers})"
                            f" [{chunk_result.chunk_idx + 1}/{total} 完了]"
                        ),
                    )
