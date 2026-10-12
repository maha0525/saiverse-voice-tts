"""The speech synthesis server over real localhost HTTP, with a fake engine.

Design: docs/intent/synthesis_server.md (verification 2 and 3). No model, no
GPU, no SAIVerse data; profiles live in a temporary directory.
"""
from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import unittest
import wave
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fake_synthesis_engine import SAMPLE_RATE, FakeEngine, chunk_audio  # noqa: E402
from synthesis_server_harness import (  # noqa: E402
    RunningServer,
    ipc,
    make_audio,
    read_frames,
    synthesis_server,
)


def _old_server_pcm(chunks):
    """What the previous external server returned: concatenated, rint(x*32768), clipped."""
    audio = np.concatenate(chunks).astype(np.float64)
    return np.clip(np.rint(audio * 32768.0), -32768, 32767).astype("<i2").tobytes()


class SynthesisServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.srv = RunningServer(self.engine).__enter__()
        self.addCleanup(self.srv.__exit__, None, None, None)
        self.assertEqual(self.srv.register("persona_1").json(), {"ok": True})

    def _post(self, body, **kwargs):
        return requests.post(f"{self.srv.url}/synthesize", json=body, timeout=10, **kwargs)

    # -- /health and profiles ------------------------------------------

    def test_health_keeps_old_fields_and_adds_engine_and_protocol(self):
        body = requests.get(f"{self.srv.url}/health", timeout=5).json()
        self.assertEqual(body, {
            "ok": True, "model_loaded": True, "device": "fake-device",
            "engine": "gpt_sovits", "protocol": 2,
        })

    def test_profile_lookup_and_storage_layout(self):
        missing = requests.get(f"{self.srv.url}/profiles/nobody", timeout=5)
        self.assertEqual((missing.status_code, missing.json()), (404, {"exists": False}))
        found = requests.get(f"{self.srv.url}/profiles/persona_1", timeout=5)
        self.assertEqual(found.json(), {"exists": True, "ref_text": "参照音声です"})

        profile_dir = self.srv.data_dir / "profiles" / "persona_1"
        stored = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["ref_text"], "参照音声です")
        self.assertRegex(stored["ref_audio"], r"^ref_[0-9a-f]{16}\.wav$")
        self.assertEqual((profile_dir / "ref_text.txt").read_text(encoding="utf-8"), "参照音声です")
        self.assertEqual((profile_dir / stored["ref_audio"]).read_bytes(), make_audio(4.0))

        # Replacing the reference changes the path and removes the old file
        self.srv.register("persona_1", ref_text="新しい書き起こし", audio=make_audio(5.0))
        replaced = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
        self.assertNotEqual(replaced["ref_audio"], stored["ref_audio"])
        self.assertEqual(sorted(p.name for p in profile_dir.glob("ref_*")),
                         sorted([replaced["ref_audio"], "ref_text.txt"]))

    def test_compressed_reference_registers_with_its_extension(self):
        writable = sf.available_formats()
        fmt, ext = ("MP3", ".mp3") if "MP3" in writable else ("OGG", ".ogg")
        try:
            audio = make_audio(4.0, fmt=fmt)
        except Exception:  # libsndfile without that encoder
            fmt, ext, audio = "FLAC", ".flac", make_audio(4.0, fmt="FLAC")
        response = self.srv.register("persona_mp3", audio=audio, filename=f"voice{ext.upper()}")
        self.assertEqual((response.status_code, response.json()), (200, {"ok": True}), fmt)
        stored = json.loads(
            (self.srv.data_dir / "profiles" / "persona_mp3" / "profile.json").read_text(encoding="utf-8")
        )
        self.assertTrue(stored["ref_audio"].endswith(ext), stored)
        self.assertEqual(self._post({"profile_id": "persona_mp3", "text": "hello"}).status_code, 200)
        self.assertTrue(self.engine.calls[-1]["ref_audio"].endswith(ext))

    def test_rejected_references(self):
        cases = [
            ("short", {"audio": make_audio(2.0)}, "2.00 s"),
            ("long", {"audio": make_audio(11.0)}, "11.00 s"),
            ("garbage", {"audio": b"not audio at all" * 10}, "not a readable audio"),
            ("empty_text", {"ref_text": "   "}, "ref_text is empty"),
        ]
        for profile_id, kwargs, reason in cases:
            with self.subTest(profile_id):
                response = self.srv.register(profile_id, **kwargs)
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.json()["ok"])
                self.assertIn(reason, response.json()["error"])
                missing = requests.get(f"{self.srv.url}/profiles/{profile_id}", timeout=5)
                self.assertEqual(missing.status_code, 404)
                self.assertFalse((self.srv.data_dir / "profiles" / profile_id).exists())
        # nothing half-written is left behind
        self.assertEqual(list((self.srv.data_dir / "profiles").glob(".upload_*")), [])

    def test_bad_profile_id_and_oversized_upload(self):
        # Rejected before reading the upload; the connection stays usable
        with requests.Session() as session:
            for _ in range(3):
                response = session.put(
                    f"{self.srv.url}/profiles/bad.id",
                    files={"ref_audio": ("ref.wav", make_audio(4.0))}, data={"ref_text": "x"},
                    timeout=10,
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("profile_id", response.json()["error"])
            self.assertEqual(session.get(f"{self.srv.url}/nowhere", timeout=5).status_code, 404)
            self.assertEqual(session.post(f"{self.srv.url}/nowhere", json={}, timeout=5).status_code, 404)
            self.assertTrue(session.get(f"{self.srv.url}/health", timeout=5).json()["ok"])
        # The 50MB limit is checked from Content-Length before reading the body
        with socket.create_connection(("127.0.0.1", self.srv.server.server_address[1]), timeout=5) as sock:
            sock.sendall(
                b"PUT /profiles/persona_1 HTTP/1.1\r\nHost: x\r\n"
                b"Content-Type: multipart/form-data; boundary=x\r\n"
                + f"Content-Length: {synthesis_server.MAX_UPLOAD_BYTES + 1}\r\n\r\n".encode()
            )
            status_line = sock.makefile("rb").readline()
        self.assertIn(b" 413 ", status_line)

    # -- /synthesize ----------------------------------------------------

    def test_old_style_request_returns_the_same_wav_as_the_old_server(self):
        """A client that only sends profile_id + text gets one 16-bit mono WAV."""
        response = self._post({"profile_id": "persona_1", "text": "こんにちは"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "audio/wav")
        with wave.open(io.BytesIO(response.content)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()),
                             (1, 2, SAMPLE_RATE))
            pcm = wav.readframes(wav.getnframes())
        self.assertEqual(pcm, _old_server_pcm([chunk_audio(i, 240) for i in range(3)]))
        call = self.engine.calls[-1]
        self.assertEqual(call["params"], {})  # engine defaults, as the old server
        self.assertEqual(call["ref_text"], "参照音声です")
        self.assertTrue(Path(call["ref_audio"]).is_absolute())
        self.assertTrue(Path(call["ref_audio"]).is_file())

    def test_stream_returns_frames_in_order_and_params_reach_engine(self):
        params = {"speed": 1.25, "top_k": 7, "chunks": 4}
        response = self._post(
            {"profile_id": "persona_1", "text": "hello", "params": params, "stream": True},
            stream=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/octet-stream")
        frames = read_frames(response.raw)
        kinds = [ipc.decode_response(f)[0] for f in frames]
        self.assertEqual(kinds, ["chunk"] * 4 + ["end"])
        for index, frame in enumerate(frames[:4]):
            audio, sr = ipc.decode_response(frame)[1]
            self.assertEqual(sr, SAMPLE_RATE)
            np.testing.assert_array_equal(audio, chunk_audio(index, 240))
        self.assertEqual(self.engine.calls[-1]["params"], params)
        # params also apply to a WAV (non-stream) request
        self._post({"profile_id": "persona_1", "text": "hello", "params": {"speed": 0.8}})
        self.assertEqual(self.engine.calls[-1]["params"], {"speed": 0.8})

    def test_errors_before_the_first_frame_are_status_codes(self):
        cases = [
            ({"profile_id": "bad id", "text": "x"}, 400),
            ({"profile_id": "persona_1", "text": "   "}, 400),
            ({"profile_id": "persona_1", "text": "x", "params": [1]}, 400),
            ({"profile_id": "nobody", "text": "x"}, 404),
            ({"profile_id": "persona_1", "text": "BOOM"}, 500),
        ]
        for body, status in cases:
            for stream in (False, True):
                with self.subTest(body=body, stream=stream):
                    response = self._post(dict(body, stream=stream))
                    self.assertEqual(response.status_code, status)
                    self.assertEqual(response.headers["Content-Type"], "application/json; charset=utf-8")
                    self.assertFalse(response.json()["ok"])
                    self.assertTrue(response.json()["error"])
        self.assertIn("fake engine failure", self._post({"profile_id": "persona_1", "text": "BOOM"}).json()["error"])

    def test_error_after_the_first_frame_is_an_error_frame_then_end(self):
        response = self._post(
            {"profile_id": "persona_1", "text": "MIDFAIL", "stream": True}, stream=True,
        )
        self.assertEqual(response.status_code, 200)
        decoded = [ipc.decode_response(f) for f in read_frames(response.raw)]
        self.assertEqual([k for k, _ in decoded], ["chunk", "error", "end"])
        self.assertIn("fake failure mid-stream", decoded[1][1])
        # The non-stream form reports the same failure as a status code
        wav = self._post({"profile_id": "persona_1", "text": "MIDFAIL"})
        self.assertEqual(wav.status_code, 500)

    def test_client_disconnect_stops_synthesis_and_frees_the_slot(self):
        self.engine.delay = 0.01
        response = self._post(
            {"profile_id": "persona_1", "text": "endless", "stream": True,
             "params": {"chunks": 100000, "samples": 24000}},
            stream=True,
        )
        first = ipc.read_frame(response.raw)
        self.assertEqual(ipc.decode_response(first)[0], "chunk")
        self.engine.finished_event.clear()
        response.close()
        self.assertTrue(self.engine.finished_event.wait(10), "synthesis kept running")
        text, reason, produced = self.engine.finished[-1]
        self.assertEqual((text, reason), ("endless", "closed"))
        self.assertLess(produced, 100000)
        self.engine.delay = 0.0
        self.assertEqual(self._post({"profile_id": "persona_1", "text": "next"}).status_code, 200)

    def test_client_that_stops_reading_is_cut_off_by_the_write_timeout(self):
        self.srv.service.write_timeout = 1.0
        body = json.dumps({
            "profile_id": "persona_1", "text": "flood", "stream": True,
            "params": {"chunks": 100000, "samples": 1_000_000},
        }).encode()
        sock = socket.create_connection(("127.0.0.1", self.srv.server.server_address[1]), timeout=5)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            sock.sendall(
                b"POST /synthesize HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
            )
            started = time.monotonic()
            self.assertTrue(self.engine.finished_event.wait(30), "slot held by a client that never reads")
            self.assertEqual(self.engine.finished[-1][:2], ("flood", "closed"))
            self.assertLess(time.monotonic() - started, 30)
            self.assertEqual(self._post({"profile_id": "persona_1", "text": "next"}).status_code, 200)
        finally:
            sock.close()

    def test_synthesis_is_serialized(self):
        """Two concurrent requests never run on the engine at the same time."""
        self.engine.delay = 0.05
        active = []
        peak = []
        original = self.engine.synthesize_stream

        def tracking(*args, **kwargs):
            active.append(1)
            peak.append(len(active))
            try:
                yield from original(*args, **kwargs)
            finally:
                active.pop()

        self.engine.synthesize_stream = tracking
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(
                self._post({"profile_id": "persona_1", "text": "x"}).status_code))
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(results, [200, 200, 200])
        self.assertEqual(max(peak), 1)


class LoadingServerTests(unittest.TestCase):
    def test_503_while_the_model_loads_then_synthesis(self):
        gate = threading.Event()
        engine = FakeEngine(load_gate=gate)
        with RunningServer(engine, wait_loaded=False) as srv:
            health = requests.get(f"{srv.url}/health", timeout=5).json()
            self.assertEqual((health["ok"], health["model_loaded"], health["protocol"]), (True, False, 2))
            # Profiles can be registered while the model loads
            self.assertEqual(srv.register("persona_1").status_code, 200)
            started = time.monotonic()
            for stream in (False, True):
                response = requests.post(
                    f"{srv.url}/synthesize",
                    json={"profile_id": "persona_1", "text": "x", "stream": stream}, timeout=5,
                )
                self.assertEqual(response.status_code, 503)
                self.assertIn("loading", response.json()["error"])
            self.assertLess(time.monotonic() - started, 2.0)  # not made to wait
            gate.set()
            srv.wait_loaded()
            ok = requests.post(f"{srv.url}/synthesize",
                               json={"profile_id": "persona_1", "text": "x"}, timeout=5)
            self.assertEqual(ok.status_code, 200)

    def test_failed_load_is_reported(self):
        def broken():
            raise RuntimeError("no weights")

        with RunningServer(None, factory=broken, wait_loaded=False) as srv:
            deadline = time.monotonic() + 5
            while srv.service.load_error is None and time.monotonic() < deadline:
                time.sleep(0.01)
            health = requests.get(f"{srv.url}/health", timeout=5).json()
            self.assertFalse(health["ok"])
            self.assertIn("no weights", health["load_error"])
            srv.register("persona_1")
            response = requests.post(f"{srv.url}/synthesize",
                                     json={"profile_id": "persona_1", "text": "x"}, timeout=5)
            self.assertEqual(response.status_code, 503)
            self.assertIn("no weights", response.json()["error"])


class EngineConfigTests(unittest.TestCase):
    def test_cli_config_overrides_addon_defaults(self):
        config = synthesis_server.load_engine_config("irodori", {"device": "cpu", "extra": 1})
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["extra"], 1)
        # the rest comes from config/default.json (or its template)
        self.assertIn("checkpoint", config)

    def test_data_dir_is_made_absolute(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                service = synthesis_server.SynthesisService(lambda: FakeEngine(), Path("rel_data"))
            finally:
                os.chdir(previous)
            self.assertTrue(service.data_dir.is_absolute())
            self.assertEqual(service.data_dir, (Path(tmp) / "rel_data").resolve())


if __name__ == "__main__":
    unittest.main()
