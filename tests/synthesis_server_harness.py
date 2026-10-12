"""Runs the real speech synthesis server (tools/speak/server.py) on localhost
with a fake engine, for the server and remote client tests. No model, no GPU,
no SAIVerse data: profiles go to a temporary directory.
"""
from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import requests
import soundfile as sf

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak import server as synthesis_server  # noqa: E402
from tools.speak.engine import subprocess_ipc as ipc  # noqa: E402


def make_audio(seconds: float, sample_rate: int = 16000, fmt: str = "WAV",
               subtype: Optional[str] = None) -> bytes:
    """A reference recording of ``seconds`` seconds in ``fmt`` (soundfile format)."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


def read_frames(raw: Any) -> List[bytes]:
    """All length-prefixed frames of a streamed body, until EOF."""
    frames = []
    while True:
        payload = ipc.read_frame(raw)
        if payload is None:
            return frames
        frames.append(payload)


class RunningServer:
    """``with RunningServer(engine) as srv:`` → ``srv.url`` is the live server."""

    def __init__(self, engine: Any, *, write_timeout: float = 5.0,
                 factory: Optional[Callable[[], Any]] = None, wait_loaded: bool = True):
        self.engine = engine
        self._factory = factory or (lambda: engine)
        self._write_timeout = write_timeout
        self._wait_loaded = wait_loaded

    def __enter__(self) -> "RunningServer":
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.server = synthesis_server.create_server(
            self._factory, self.data_dir, host="127.0.0.1", port=0,
            write_timeout=self._write_timeout,
        )
        self.service = self.server.service
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        )
        self._thread.start()
        if self._wait_loaded:
            self.wait_loaded()
        return self

    def wait_loaded(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.service.model_loaded:
                return
            time.sleep(0.01)
        raise AssertionError("fake model did not load")

    def register(self, profile_id: str, ref_text: str = "参照音声です",
                 audio: Optional[bytes] = None, filename: str = "ref.wav") -> requests.Response:
        if audio is None:
            audio = make_audio(4.0)
        return requests.put(
            f"{self.url}/profiles/{profile_id}",
            files={"ref_audio": (filename, audio, "application/octet-stream")},
            data={"ref_text": ref_text}, timeout=10,
        )

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)
        self._tmp.cleanup()
