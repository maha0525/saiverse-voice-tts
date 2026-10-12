"""The shared synthesis function (engine/synthesis_core) and its two callers.

Design: docs/intent/synthesis_server.md (verification 1). The real
subprocess_worker runs as a child process with a fake engine, and the real
server runs on localhost with the same fake engine; both must send the same
frames in the same order for the same request.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import requests

_TESTS = Path(__file__).resolve().parent
_PACK_ROOT = _TESTS.parent
for _path in (_PACK_ROOT, _TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fake_synthesis_engine import SAMPLE_RATE, FakeEngine, chunk_audio  # noqa: E402
from synthesis_server_harness import RunningServer, read_frames  # noqa: E402
from tools.speak.engine import subprocess_ipc as ipc  # noqa: E402
from tools.speak.engine.synthesis_core import (  # noqa: E402
    create_local_engine,
    iter_response_frames,
)

_SPEAK_DIR = _PACK_ROOT / "tools" / "speak"

# Runs the real subprocess_worker.main() with the fake engine in place of the
# real GPT-SoVITS (the worker resolves create_local_engine at call time).
_WORKER_WRAPPER = '''
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from engine import synthesis_core
from fake_synthesis_engine import FakeEngine
synthesis_core.create_local_engine = lambda name, config: FakeEngine()
sys.argv = ["subprocess_worker.py", "--engine", "gpt_sovits", "--config", "{}"]
from engine import subprocess_worker
sys.exit(subprocess_worker.main())
'''


def _request(method="stream", text="hello", params=None):
    return {"method": method, "text": text, "ref_audio": "/ref.wav", "ref_text": "ref", "params": params or {}}


class SynthesisCoreTests(unittest.TestCase):
    def test_stream_frames_then_end(self):
        frames = list(iter_response_frames(FakeEngine(), _request()))
        decoded = [ipc.decode_response(f) for f in frames]
        self.assertEqual([k for k, _ in decoded], ["chunk", "chunk", "chunk", "end"])
        for index, (_kind, (audio, sr)) in enumerate(decoded[:3]):
            self.assertEqual(sr, SAMPLE_RATE)
            np.testing.assert_array_equal(audio, chunk_audio(index, 240))

    def test_synthesize_method_returns_one_result(self):
        decoded = [ipc.decode_response(f) for f in iter_response_frames(FakeEngine(), _request("synthesize"))]
        self.assertEqual([k for k, _ in decoded], ["result", "end"])
        audio, sr, duration = decoded[0][1]
        self.assertEqual((len(audio), sr, duration), (720, SAMPLE_RATE, 30))

    def test_params_and_reference_reach_the_engine(self):
        engine = FakeEngine()
        list(iter_response_frames(engine, _request(params={"speed": 1.5})))
        self.assertEqual(engine.calls[-1], {
            "method": "stream", "text": "hello", "ref_audio": "/ref.wav",
            "ref_text": "ref", "params": {"speed": 1.5},
        })

    def test_engine_failures_become_error_then_end(self):
        for text, kinds in (("BOOM", ["error", "end"]), ("MIDFAIL", ["chunk", "error", "end"])):
            with self.subTest(text=text):
                decoded = [ipc.decode_response(f) for f in iter_response_frames(FakeEngine(), _request(text=text))]
                self.assertEqual([k for k, _ in decoded], kinds)
                self.assertTrue(decoded[-2][1].startswith("RuntimeError: fake"))

    def test_closing_stops_the_engine(self):
        engine = FakeEngine(chunks=1000)
        frames = iter_response_frames(engine, _request())
        next(frames)
        frames.close()
        self.assertEqual(engine.finished, [("hello", "closed", 0)])

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):
            create_local_engine("openai_tts", {})


class ChildAndServerParityTests(unittest.TestCase):
    """The child process and the server send identical frames for one request."""

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix="_worker_wrapper.py")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_WORKER_WRAPPER)
        self.addCleanup(os.unlink, path)
        self.proc = subprocess.Popen(
            [sys.executable, path, str(_SPEAK_DIR), str(_TESTS)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        self.addCleanup(self._stop_child)

    def _stop_child(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()

    def _child_frames(self, text, params):
        ipc.write_frame(self.proc.stdin, ipc.encode_request("stream", text, "/ref.wav", "ref", params))
        frames = []
        while True:
            payload = ipc.read_frame(self.proc.stdout)
            self.assertIsNotNone(payload, "worker exited")
            frames.append(payload)
            if payload[:1] == b"E":
                return frames

    def test_same_frames_in_the_same_order(self):
        with RunningServer(FakeEngine()) as srv:
            srv.register("persona_1")
            for text, params in (("hello", {}), ("params", {"chunks": 5, "samples": 100}),
                                 ("MIDFAIL", {}), ("BOOM", {})):
                with self.subTest(text=text):
                    child = self._child_frames(text, params)
                    response = requests.post(
                        f"{srv.url}/synthesize",
                        json={"profile_id": "persona_1", "text": text, "params": params, "stream": True},
                        stream=True, timeout=10,
                    )
                    if text == "BOOM":
                        # Before the first frame the server answers with a status code
                        # and the same reason the child sends in its error frame.
                        self.assertEqual(response.status_code, 500)
                        self.assertEqual([p[:1] for p in child], [b"X", b"E"])
                        self.assertEqual(response.json()["error"], ipc.decode_response(child[0])[1])
                        continue
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(read_frames(response.raw), child)
        # The child process kept serving after the failures
        self.assertEqual(len(self._child_frames("again", {})), 4)


if __name__ == "__main__":
    unittest.main()
