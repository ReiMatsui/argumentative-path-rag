"""
OpenAI Embeddings を SentenceTransformer 互換のインターフェースで提供する
アダプタ。

目的:
  * M1 Mac / CPU 環境で E5-Mistral-7B (14GB) のロードを避ける
  * sentence-transformers / torch の重依存を回避
  * 標準的な「dense RAG の強いベースライン」として OpenAI embeddings を使う

使い方:
    from openai import OpenAI
    from ap_rag.retrieval.openai_encoder import OpenAIEncoder

    client = OpenAI()
    encoder = OpenAIEncoder(client=client, model="text-embedding-3-small")

    # EmbeddingNodeSelector に注入
    selector = EmbeddingNodeSelector(model_name="openai", device="cpu")
    selector._model = encoder   # 遅延ロードをバイパス

    # DenseRAG に注入 (encoder 引数経由)
    dense = DenseRAG(client=client, embedding_model="openai", encoder=encoder)

本アダプタは `.encode(texts, ...)` の最小サブセットを提供する:
  * `texts: str | list[str]`
  * `convert_to_numpy: bool = True`   （常に numpy を返す。tensor は未サポート）
  * `normalize_embeddings: bool = True`
  * `batch_size: int | None`
  * `show_progress_bar: bool = False`  （互換のため受け取るだけ）
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OpenAIEncoder:
    """OpenAI Embeddings API を SentenceTransformer 互換で叩くアダプタ。

    Args:
        client: openai.OpenAI インスタンス。
        model: embeddings モデル名。text-embedding-3-small (1536 dim) を推奨。
        batch_size: 1 回の API 呼び出しに乗せるテキスト数のデフォルト値。
    """

    # OpenAI embeddings の上限は最大 2048 入力 / 呼び出し（公式ドキュメント）
    _MAX_INPUTS_PER_CALL = 256

    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
        batch_size: int = 128,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = min(max(1, batch_size), self._MAX_INPUTS_PER_CALL)

    def encode(
        self,
        texts: Any,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = False,
        **_: Any,
    ) -> np.ndarray:
        """テキストを埋め込みベクトルに変換する。

        戻り値は常に numpy.ndarray。単一文字列なら 1D、リストなら 2D。
        convert_to_tensor=True を渡された場合は内部で numpy を torch.tensor に
        変換する（呼び出し側がその型を期待する場合の互換性のため）。
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)
        if not text_list:
            arr = np.zeros((0, 1), dtype=np.float32)
            return arr[0] if is_single else arr

        bs = batch_size or self._batch_size
        bs = min(bs, self._MAX_INPUTS_PER_CALL)

        all_vecs: list[list[float]] = []
        for i in range(0, len(text_list), bs):
            batch = text_list[i : i + bs]
            # 空文字は API エラーになるので一時的に空白1文字に置き換える
            safe_batch = [t if t else " " for t in batch]
            response = self._client.embeddings.create(
                model=self._model,
                input=safe_batch,
            )
            # OpenAI SDK は data を index 順で返すが、念のためソート
            data = sorted(response.data, key=lambda d: d.index)
            all_vecs.extend(d.embedding for d in data)

        arr = np.asarray(all_vecs, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            arr = arr / norms

        if convert_to_tensor:
            # DenseRAG の既存コードパスのために提供（torch がなければ numpy のまま返す）
            try:
                import torch

                tensor = torch.from_numpy(arr)
                return tensor[0] if is_single else tensor  # type: ignore[return-value]
            except ImportError:
                logger.warning(
                    "convert_to_tensor=True が要求されましたが torch がないため "
                    "numpy 配列を返します。呼び出し側が numpy に対応しているか確認してください。"
                )

        if is_single:
            return arr[0]
        return arr
