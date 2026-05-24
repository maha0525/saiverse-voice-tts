"""本体プロセスと音声合成子プロセスの間の通信フレーム。

標準入出力 (バイナリ) 上に「4 バイトのビッグエンディアン長 + ペイロード」の
フレームを流す。ペイロード先頭 1 バイトを種別タグとする。

- リクエスト (本体 -> 子): ``b"Q"`` + UTF-8 JSON
    ``{"method": "stream"|"synthesize", "text", "ref_audio", "ref_text", "params"}``
- 応答 (子 -> 本体):
    - ``b"C"`` + 4 バイト sample_rate(BE) + float32 little-endian 音声バイト列  (チャンク)
    - ``b"R"`` + 4 バイト sample_rate(BE) + 4 バイト duration_ms(BE) + float32 バイト列 (一括結果)
    - ``b"E"``  (このリクエストの応答終わり)
    - ``b"X"`` + UTF-8 エラーメッセージ

音声は ``np.float32`` を ``<f4`` (little-endian) の生バイトで送り、受け側で
``np.frombuffer(buf, dtype="<f4")`` で復元する。base64 を挟まないので軽い。
"""
from __future__ import annotations

import json
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np

_LEN = struct.Struct(">I")


def write_frame(stream, payload: bytes) -> None:
    """長さ付きフレームを書いて flush する。"""
    stream.write(_LEN.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, n: int) -> Optional[bytes]:
    """ちょうど ``n`` バイト読む。EOF なら None。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_frame(stream) -> Optional[bytes]:
    """1 フレーム読む。EOF (相手プロセス終了) なら None。"""
    header = _read_exact(stream, _LEN.size)
    if header is None:
        return None
    (length,) = _LEN.unpack(header)
    if length == 0:
        return b""
    return _read_exact(stream, length)


# --- request -------------------------------------------------------------

def encode_request(
    method: str,
    text: str,
    ref_audio: Optional[str],
    ref_text: Optional[str],
    params: Optional[Dict[str, Any]],
) -> bytes:
    body = json.dumps(
        {
            "method": method,
            "text": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "params": params or {},
        }
    ).encode("utf-8")
    return b"Q" + body


def decode_request(payload: bytes) -> Dict[str, Any]:
    if not payload or payload[:1] != b"Q":
        raise ValueError("not a request frame")
    return json.loads(payload[1:].decode("utf-8"))


# --- responses -----------------------------------------------------------

def encode_chunk(audio: np.ndarray, sample_rate: int) -> bytes:
    return b"C" + _LEN.pack(int(sample_rate)) + audio.astype("<f4", copy=False).tobytes()


def encode_result(audio: np.ndarray, sample_rate: int, duration_ms: int) -> bytes:
    return (
        b"R"
        + _LEN.pack(int(sample_rate))
        + _LEN.pack(int(duration_ms))
        + audio.astype("<f4", copy=False).tobytes()
    )


def encode_end() -> bytes:
    return b"E"


def encode_error(message: str) -> bytes:
    return b"X" + message.encode("utf-8", errors="replace")


def decode_response(payload: bytes) -> Tuple[str, Any]:
    """応答フレームを (種別, 値) に分解する。

    返り値:
      - ("chunk", (audio: np.ndarray, sample_rate: int))
      - ("result", (audio: np.ndarray, sample_rate: int, duration_ms: int))
      - ("end", None)
      - ("error", message: str)
    """
    if not payload:
        raise ValueError("empty response frame")
    tag = payload[:1]
    if tag == b"C":
        sr = _LEN.unpack(payload[1:5])[0]
        audio = np.frombuffer(payload[5:], dtype="<f4")
        return "chunk", (audio, sr)
    if tag == b"R":
        sr = _LEN.unpack(payload[1:5])[0]
        dur = _LEN.unpack(payload[5:9])[0]
        audio = np.frombuffer(payload[9:], dtype="<f4")
        return "result", (audio, sr, dur)
    if tag == b"E":
        return "end", None
    if tag == b"X":
        return "error", payload[1:].decode("utf-8", errors="replace")
    raise ValueError(f"unknown response tag: {tag!r}")
