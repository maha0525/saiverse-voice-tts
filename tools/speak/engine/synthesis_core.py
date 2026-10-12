"""ローカル合成エンジンを回して、声の断片を通信フレームとして返す共通の合成関数。

子プロセス (subprocess_worker、標準入出力でつなぐ) と音声合成サーバー
(tools/speak/server.py、HTTP でつなぐ) の両方がここを呼ぶ。エンジンを回す
コードはこの一箇所にだけ置く (docs/intent/synthesis_server.md「共通の合成関数」)。

関数は書き出し先を持たず、送るべきフレーム (subprocess_ipc の encode_* が返す
バイト列) を順に返す。書き出し (標準出力 / HTTP の応答本文) は呼び出し側の仕事。

このモジュールはエンジン専用の Python 環境の中で動く。SAIVerse 本体を import
しない。
"""
from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, Iterator

from .base import TTSEngine
from .subprocess_ipc import encode_chunk, encode_end, encode_error, encode_result

# 共通の合成関数で動かすローカル合成エンジン
LOCAL_ENGINES = ("gpt_sovits", "irodori")


def create_local_engine(name: str, config: Dict[str, Any]) -> TTSEngine:
    """ローカル合成エンジンの本物を作る (代理エンジンではない)。

    子プロセスとサーバーの中で使う。本体プロセスの create_engine は代理エンジン
    を返すので、ここで呼ぶと再帰になる — だから本物のクラスを直接作る。
    """
    if name == "irodori":
        from .irodori import IrodoriEngine

        return IrodoriEngine(config)
    if name == "gpt_sovits":
        from .gpt_sovits import GPTSoVITSEngine

        return GPTSoVITSEngine(config)
    raise ValueError(f"unknown local synthesis engine: {name!r}")


def iter_response_frames(engine: TTSEngine, request: Dict[str, Any]) -> Iterator[bytes]:
    """合成の依頼を一つ処理し、応答のフレームを順に返す。

    ``request`` は ``{"method": "stream"|"synthesize", "text", "ref_audio",
    "ref_text", "params"}`` (subprocess_ipc.decode_request の返す形)。

    返すフレーム:
      - method="stream": 断片 (``C``) を順に、最後に終了 (``E``)
      - method="synthesize": 一括の結果 (``R``)、最後に終了 (``E``)
      - エンジンが失敗したら: そこまでの断片の後にエラー (``X``) と終了 (``E``)

    呼び出し側が途中でこのジェネレータを閉じると (書き出しの失敗など)、
    エンジンのジェネレータも閉じて合成を打ち切る。
    """
    method = request.get("method", "stream")
    text = request.get("text", "")
    ref_audio = request.get("ref_audio")
    ref_text = request.get("ref_text")
    params = request.get("params") or {}
    try:
        if method == "synthesize":
            result = engine.synthesize(text, ref_audio, ref_text, params)
            yield encode_result(result.audio, result.sample_rate, result.duration_ms)
        else:
            stream = engine.synthesize_stream(text, ref_audio, ref_text, params)
            try:
                for chunk in stream:
                    yield encode_chunk(chunk.audio, chunk.sample_rate)
            finally:
                # GPT-SoVITS は合成の間だけ作業フォルダを移す (_cwd)。閉じれば
                # その場で元に戻り、推論も止まる。
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
    except Exception as exc:  # 1 件の失敗で呼び出し側 (プロセス / サーバー) は落とさない
        traceback.print_exc(file=sys.stderr)
        yield encode_error(f"{type(exc).__name__}: {exc}")
    yield encode_end()
