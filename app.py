"""
Argumentative-Path RAG — Web Visualization UI

グラフ構築・クエリ実行・探索経路を可視化するWebアプリ。

起動:
    uv run python app.py
    # → http://localhost:8000 をブラウザで開く

環境変数:
    OPENAI_API_KEY  (.env から自動ロード)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Argumentative-Path RAG Visualizer", version="0.1.0")

# ── グローバル状態 ─────────────────────────────────────────────────────────────
# プロセス内でグラフ・パイプラインを保持する

_client: Any = None
_store: Any = None
_rag_pipeline: Any = None
_bm25: Any = None
_indexed_graphs: dict[str, Any] = {}   # doc_id → ArgumentGraph


# ── 初期化ヘルパー ─────────────────────────────────────────────────────────────

def _get_client():
    global _client
    if _client is None:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY が設定されていません")
        _client = openai.OpenAI(api_key=api_key)
    return _client


def _get_store():
    global _store
    if _store is None:
        from ap_rag.graph.networkx_store import NetworkXGraphStore
        _store = NetworkXGraphStore()
    return _store


def _get_pipeline():
    global _rag_pipeline
    if _rag_pipeline is None:
        from ap_rag.pipeline import ArgumentativeRAGPipeline
        from ap_rag.retrieval.context_builder import ContextBuilder
        from ap_rag.retrieval.query_classifier import QueryClassifier
        from ap_rag.retrieval.traversal import GraphTraverser
        from ap_rag.generation.generator import AnswerGenerator
        client = _get_client()
        store = _get_store()
        _rag_pipeline = ArgumentativeRAGPipeline(
            store=store,
            query_classifier=QueryClassifier(client=client, model="gpt-4o-mini"),
            traverser=GraphTraverser(store=store),
            context_builder=ContextBuilder(max_nodes=15),
            generator=AnswerGenerator(client=client, model="gpt-4o-mini"),
        )
    return _rag_pipeline


# ── リクエスト/レスポンスモデル ────────────────────────────────────────────────

class IndexRequest(BaseModel):
    text: str
    doc_id: str = "demo"


class QueryRequest(BaseModel):
    question: str
    doc_id: str = "demo"


# ── BFS ステップ追跡（可視化用） ───────────────────────────────────────────────

def _compute_traversal_steps(store, doc_id: str, query_type) -> list[dict]:
    """探索戦略に基づいてBFSを再現し、ステップごとのノードIDを返す。"""
    from ap_rag.models.taxonomy import TRAVERSAL_STRATEGIES

    strategy = TRAVERSAL_STRATEGIES[query_type]
    edges_by_dir = strategy.edges_by_direction()

    # 入口ノード
    entry_nodes: list[Any] = []
    for nt in strategy.entry_node_types:
        entry_nodes.extend(store.get_nodes_by_type(doc_id, nt))

    if not entry_nodes:
        return []

    steps = [{
        "step": 0,
        "label": "入口ノード（" + " / ".join(n.value for n in strategy.entry_node_types) + "）",
        "node_ids": [n.id for n in entry_nodes],
    }]

    visited: set[str] = {n.id for n in entry_nodes}
    current_level = entry_nodes

    for depth in range(1, strategy.max_depth + 1):
        next_seen: set[str] = set()
        next_level: list[Any] = []

        for node in current_level:
            for direction, edge_vals in edges_by_dir.items():
                if not edge_vals:
                    continue
                neighbors = store.get_neighbors(
                    node.id, edge_types=edge_vals, direction=direction
                )
                for nbr in neighbors:
                    if (nbr.id not in visited
                            and nbr.id not in next_seen
                            and nbr.node_type not in strategy.exclude_node_types):
                        next_level.append(nbr)
                        next_seen.add(nbr.id)

        if not next_level:
            break

        steps.append({
            "step": depth,
            "label": f"深さ {depth}（{len(next_level)} ノード展開）",
            "node_ids": [n.id for n in next_level],
        })
        visited.update(next_seen)
        current_level = next_level

    return steps


# ── API エンドポイント ─────────────────────────────────────────────────────────

@app.post("/api/index")
async def index_document(req: IndexRequest):
    """文書をインデックスしてグラフを構築する。"""
    global _bm25

    from ap_rag.indexing.chunker import SentenceChunker
    from ap_rag.indexing.classifier import NodeClassifier
    from ap_rag.indexing.extractor import EdgeExtractor
    from ap_rag.indexing.pipeline import IndexingPipeline
    from ap_rag.evaluation.baselines import BM25RAG

    client = _get_client()
    store = _get_store()

    indexer = IndexingPipeline(
        chunker=SentenceChunker(),
        classifier=NodeClassifier(client=client, model="gpt-4o-mini"),
        extractor=EdgeExtractor(client=client, model="gpt-4o-mini"),
        store=store,
        show_progress=False,
    )

    result = indexer.run(req.text, doc_id=req.doc_id)
    _indexed_graphs[req.doc_id] = result.graph

    # BM25 インデックス
    chunker = SentenceChunker()
    chunks = chunker.chunk(req.text, doc_id=req.doc_id)
    _bm25 = BM25RAG(client=client, top_k=5)
    _bm25.index([c.text for c in chunks], doc_id=req.doc_id)

    stats = result.graph.stats()
    return {
        "doc_id": req.doc_id,
        "num_nodes": stats["total_nodes"],
        "num_edges": stats["total_edges"],
        "nodes_by_type": stats["nodes_by_type"],
        "num_chunks": result.num_chunks,
    }


@app.get("/api/graph/{doc_id}")
async def get_graph(doc_id: str):
    """Cytoscape.js 形式のグラフデータを返す。"""
    graph = _indexed_graphs.get(doc_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"グラフが見つかりません: {doc_id}")

    cy_nodes = [
        {
            "data": {
                "id": node.id,
                "label": node.node_type.value,
                "type": node.node_type.value,
                "text": node.text,
                "chunk_idx": node.source_chunk_idx,
            }
        }
        for node in graph.nodes.values()
    ]

    cy_edges = [
        {
            "data": {
                "id": edge.id,
                "source": edge.source_id,
                "target": edge.target_id,
                "label": edge.edge_type.value,
                "type": edge.edge_type.value,
                "confidence": round(edge.confidence, 2),
            }
        }
        for edge in graph.edges.values()
        if edge.source_id in graph.nodes and edge.target_id in graph.nodes
    ]

    return {"nodes": cy_nodes, "edges": cy_edges}


@app.post("/api/query")
async def query_document(req: QueryRequest):
    """クエリを実行し、回答・探索経路・BM25比較を返す。"""
    if req.doc_id not in _indexed_graphs:
        raise HTTPException(status_code=400, detail="先に文書をインデックスしてください")

    pipeline = _get_pipeline()
    store = _get_store()

    # ArgumentativeRAG
    result = pipeline.query(req.question, doc_id=req.doc_id)
    query_type = result.retrieval_context.query_type
    retrieved_ids = [n.id for n in result.retrieval_context.nodes]

    # BFS ステップ
    steps = _compute_traversal_steps(store, req.doc_id, query_type)

    # BM25
    bm25_answer = ""
    bm25_chunks: list[str] = []
    if _bm25:
        bm25_result = _bm25.query(req.question, doc_id=req.doc_id)
        bm25_answer = bm25_result.answer
        bm25_chunks = [n.text for n in bm25_result.retrieval_context.nodes]

    return {
        "question": req.question,
        "query_type": query_type.value,
        "answer": result.answer,
        "retrieved_node_ids": retrieved_ids,
        "traversal_steps": steps,
        "bm25_answer": bm25_answer,
        "bm25_chunks": bm25_chunks,
    }


@app.get("/api/status")
async def status():
    """サーバー状態確認。"""
    return {
        "ok": True,
        "indexed_docs": list(_indexed_graphs.keys()),
        "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }


# ── 静的ファイル配信 ───────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


# ── エントリポイント ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀  http://localhost:8000 で起動します")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
