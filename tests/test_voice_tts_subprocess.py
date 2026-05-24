"""音声合成別プロセス化 (proxy + IPC) のテスト。

実機の GPT-SoVITS モデルを使わずに検証する:
  1. 通信フレーム (subprocess_ipc) の往復エンコード/デコード。
  2. 偽 worker を起動した proxy の挙動 — ストリーム受信・エラー伝播・
     無応答時のタイムアウトと子プロセス再起動。

設計: docs/out_of_process.md
"""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

_SPEAK_DIR = _PACK_ROOT / "tools" / "speak"

from tools.speak.engine import subprocess_ipc as ipc  # noqa: E402
from tools.speak.engine import subprocess_proxy  # noqa: E402
from tools.speak.engine.subprocess_proxy import SubprocessGPTSoVITSEngine  # noqa: E402


# 偽 worker: text="BOOM" でエラー、"HANG" で無応答、それ以外は 3 チャンク返す。
# 別プロセスなので tools/speak を経路に入れて engine.subprocess_ipc を import する
# (実 worker と同じ作法)。
_FAKE_WORKER = '''
import os, sys
sys.path.insert(0, os.environ["VOICE_TTS_FAKE_IPC_DIR"])
import time
import numpy as np
from engine.subprocess_ipc import (
    read_frame, write_frame, decode_request,
    encode_chunk, encode_end, encode_error,
)
proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
proto_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
sys.stdout = sys.stderr
while True:
    p = read_frame(proto_in)
    if p is None:
        break
    req = decode_request(p)
    text = req.get("text", "")
    if text == "BOOM":
        write_frame(proto_out, encode_error("boom"))
        write_frame(proto_out, encode_end())
        continue
    if text == "HANG":
        time.sleep(30)
        continue
    for i in range(3):
        write_frame(proto_out, encode_chunk(np.array([float(i)], dtype=np.float32), 32000))
    write_frame(proto_out, encode_end())
'''


class IpcRoundTripTests(unittest.TestCase):
    def test_frame_read_write_roundtrip(self):
        buf = io.BytesIO()
        ipc.write_frame(buf, b"hello")
        ipc.write_frame(buf, b"")
        ipc.write_frame(buf, b"world!!")
        buf.seek(0)
        self.assertEqual(ipc.read_frame(buf), b"hello")
        self.assertEqual(ipc.read_frame(buf), b"")
        self.assertEqual(ipc.read_frame(buf), b"world!!")
        self.assertIsNone(ipc.read_frame(buf))  # EOF

    def test_request_roundtrip(self):
        payload = ipc.encode_request("stream", "やあ", "/ref.wav", "ref text", {"speed": 1.2})
        req = ipc.decode_request(payload)
        self.assertEqual(req["method"], "stream")
        self.assertEqual(req["text"], "やあ")
        self.assertEqual(req["ref_audio"], "/ref.wav")
        self.assertEqual(req["params"], {"speed": 1.2})

    def test_chunk_roundtrip_preserves_audio(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        kind, value = ipc.decode_response(ipc.encode_chunk(audio, 32000))
        self.assertEqual(kind, "chunk")
        got_audio, sr = value
        self.assertEqual(sr, 32000)
        np.testing.assert_array_almost_equal(got_audio, audio)

    def test_result_and_end_and_error(self):
        audio = np.array([0.1, 0.2], dtype=np.float32)
        kind, value = ipc.decode_response(ipc.encode_result(audio, 22050, 1234))
        self.assertEqual(kind, "result")
        _a, sr, dur = value
        self.assertEqual((sr, dur), (22050, 1234))
        self.assertEqual(ipc.decode_response(ipc.encode_end()), ("end", None))
        self.assertEqual(ipc.decode_response(ipc.encode_error("oops")), ("error", "oops"))


class SubprocessProxyTests(unittest.TestCase):
    def setUp(self):
        os.environ["VOICE_TTS_FAKE_IPC_DIR"] = str(_SPEAK_DIR)
        fd, path = tempfile.mkstemp(suffix="_fake_worker.py")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_FAKE_WORKER)
        self._worker_path = Path(path)
        self._orig_worker = subprocess_proxy._WORKER_PATH
        subprocess_proxy._WORKER_PATH = self._worker_path

    def tearDown(self):
        subprocess_proxy._WORKER_PATH = self._orig_worker
        try:
            self._worker_path.unlink()
        except OSError:
            pass
        os.environ.pop("VOICE_TTS_FAKE_IPC_DIR", None)

    def test_stream_yields_chunks(self):
        eng = SubprocessGPTSoVITSEngine({})
        try:
            chunks = list(eng.synthesize_stream("hello", ref_audio="/x.wav"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].sample_rate, 32000)
            np.testing.assert_array_almost_equal(chunks[2].audio, np.array([2.0], dtype=np.float32))
        finally:
            eng.close()

    def test_worker_error_propagates(self):
        eng = SubprocessGPTSoVITSEngine({})
        try:
            with self.assertRaises(RuntimeError) as ctx:
                list(eng.synthesize_stream("BOOM", ref_audio="/x.wav"))
            self.assertIn("boom", str(ctx.exception))
        finally:
            eng.close()

    def test_timeout_kills_and_restarts_worker(self):
        orig_first = subprocess_proxy._FIRST_LOAD_TIMEOUT
        orig_frame = subprocess_proxy._FRAME_TIMEOUT
        subprocess_proxy._FIRST_LOAD_TIMEOUT = 1.5
        subprocess_proxy._FRAME_TIMEOUT = 1.5
        eng = SubprocessGPTSoVITSEngine({})
        try:
            with self.assertRaises(RuntimeError):
                list(eng.synthesize_stream("HANG", ref_audio="/x.wav"))
            # 無応答でタイムアウト -> kill されている
            self.assertIsNone(eng._proc)
            # 再リクエストで自動再起動し、正常に動く
            chunks = list(eng.synthesize_stream("hello", ref_audio="/x.wav"))
            self.assertEqual(len(chunks), 3)
        finally:
            subprocess_proxy._FIRST_LOAD_TIMEOUT = orig_first
            subprocess_proxy._FRAME_TIMEOUT = orig_frame
            eng.close()


if __name__ == "__main__":
    unittest.main()
