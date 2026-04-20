"""
OpenAI API 互換ユーティリティ。

モデルによっては一部パラメータ（temperature, top_p など）に制約がある。
gpt-5 系 / o1 系 / o3 系は `temperature=1`（デフォルト）しか受け付けず、
`temperature=0.0` を渡すと 400 エラーになる。

このモジュールは **呼び出し側がモデル名を意識せずに** パラメータを渡せるよう、
「許可されるなら付ける、ダメなら黙って外す」kwargs を組み立てるヘルパを提供する。

使い方:
    resp = client.beta.chat.completions.parse(
        model=self._model,
        messages=...,
        response_format=Schema,
        **sampling_kwargs(self._model, temperature=0.0),
    )

こうしておけば gpt-4o / gpt-4.1 系ではこれまで通り temperature=0.0 が効き、
gpt-5 / o1 / o3 系では自動的に temperature が外れて 400 を避けられる。
"""

from __future__ import annotations

import os

# reasoning_effort のデフォルト値（環境変数 OPENAI_REASONING_EFFORT が無い場合）。
# 速度とコストを優先して "low" にする（default は "medium")。
#   "minimal" / "low" / "medium" / "high" のいずれか。
_DEFAULT_REASONING_EFFORT = "low"

# 有効な reasoning_effort 値（不正な値を弾くためのガード）。
_VALID_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


def _is_fixed_temperature_model(model: str) -> bool:
    """このモデルは `temperature=1` (デフォルト) 以外を受け付けない、を判定する。"""
    name = model.lower()
    # gpt-5 系（gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-turbo ... 想定の将来派生も含む）
    if name.startswith("gpt-5"):
        return True
    # 推論系モデル（o1, o1-mini, o1-preview, o3, o3-mini, o4-mini 等）
    if name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return True
    return False


def _uses_max_completion_tokens(model: str) -> bool:
    """`max_tokens` ではなく `max_completion_tokens` を要求するモデルか判定する。

    gpt-5 系 / 推論系 (o1 / o3 / o4) は `max_completion_tokens` 必須。
    """
    name = model.lower()
    return (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


def sampling_kwargs(
    model: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict:
    """モデル互換性を考慮した sampling パラメータ dict を返す。

    Args:
        model: 使用予定のモデル名。
        temperature: 指定したい temperature。モデルが非対応なら外れる。
        top_p: 指定したい top_p。同上。

    Returns:
        実際に API に渡して安全な kwargs dict。
    """
    fixed = _is_fixed_temperature_model(model)
    kwargs: dict = {}
    if temperature is not None and not fixed:
        kwargs["temperature"] = temperature
    if top_p is not None and not fixed:
        kwargs["top_p"] = top_p
    return kwargs


def max_tokens_kwarg(model: str, value: int) -> dict:
    """モデルに応じて `max_tokens` / `max_completion_tokens` を自動選択する。

    gpt-4o / gpt-4.1 系は `max_tokens`、gpt-5 系 / 推論系は `max_completion_tokens`。

    注意: gpt-5 系 / o 系は **推論 (reasoning) トークンも同じ出力枠から消費する**。
    `max_completion_tokens=64` のような小さな値を渡すと、reasoning が数十〜
    数百トークン消費した時点で枠を使い切り、可視テキスト 0 で打ち切られ
    `max_tokens reached` エラーになる (観測例: low effort でも 100 トークン超)。

    そのため推論モデルでは **最低 1024 トークン** まで自動で引き上げる。
    これは「上限 (cap)」であって事前割り当てではないので、短い応答で実際に
    消費するトークン数 (=課金対象) は増えない。推論バーストに上限を余らせて
    おく安全装置。
    """
    if _uses_max_completion_tokens(model):
        # 推論バースト用バッファ。reasoning_effort="low" で 200〜500, "medium" で
        # 500〜1500 程度まで膨らむことがあるので、安全側に倒して 1024 を floor。
        return {"max_completion_tokens": max(value, 1024)}
    return {"max_tokens": value}


def _resolve_reasoning_effort(effort: str | None) -> str:
    """引数 → 環境変数 → デフォルト の順で reasoning_effort を決める。"""
    if effort is None:
        effort = os.environ.get("OPENAI_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT)
    effort = effort.strip().lower()
    if effort not in _VALID_REASONING_EFFORTS:
        # 想定外の値 → デフォルトに戻す
        effort = _DEFAULT_REASONING_EFFORT
    return effort


def reasoning_kwarg(
    model: str,
    effort: str | None = None,
) -> dict:
    """モデルに応じた reasoning_effort kwargs を返す。

    - gpt-5 系 / o1 / o3 / o4 系: `reasoning_effort` を付ける。
    - それ以外: 空 dict（旧モデルは非対応なので渡さない）。

    優先順位:
        1. 明示引数 effort
        2. 環境変数 OPENAI_REASONING_EFFORT
        3. デフォルト "low"

    "low" は速度・コスト優先。分類タスクや JSON 抽出は "low" で十分なことが多い。
    複雑な推論が必要な判定 (LLM-as-judge など) は "medium" に上げてもよい。
    """
    if not _is_fixed_temperature_model(model):
        # 旧モデルは reasoning_effort を受け付けない
        return {}
    return {"reasoning_effort": _resolve_reasoning_effort(effort)}
