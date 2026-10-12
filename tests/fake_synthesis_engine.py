"""Test-only synthesis engine that returns fixed audio (no model, no GPU).

Shared by the synthesis core / server / launcher / remote client tests, and
imported inside a real ``subprocess_worker`` child process by
tests/test_synthesis_core.py, so it must not import the addon's packages.

Chunk ``i`` is ``samples`` float32 samples of value ``(i + 1) / 10`` at
``SAMPLE_RATE``. The text and params steer it:

- ``"BOOM"``: fails before the first chunk
- ``"MIDFAIL"``: fails after the first chunk
- params ``chunks`` / ``samples`` override the chunk count and size
- param ``gap``: seconds to pause before every chunk after the first
"""
from __future__ import annotations

import threading
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np

SAMPLE_RATE = 24000

Chunk = namedtuple("Chunk", "audio sample_rate")
Result = namedtuple("Result", "audio sample_rate duration_ms")


def chunk_audio(index: int, samples: int) -> np.ndarray:
    return np.full(samples, (index + 1) / 10.0, dtype=np.float32)


class FakeEngine:
    name = "fake"
    supports_streaming = True

    def __init__(
        self, chunks: int = 3, samples: int = 240, delay: float = 0.0,
        load_gate: Optional[threading.Event] = None,
    ):
        self.chunks = chunks
        self.samples = samples
        self.delay = delay
        self.load_gate = load_gate
        self.device: Optional[str] = None
        self.calls: List[Dict[str, Any]] = []
        # (text, "done" | "failed" | "closed", chunks produced) per synthesis
        self.finished: List[Any] = []
        self.finished_event = threading.Event()
        self.finished_at: Optional[float] = None

    def _lazy_load(self) -> None:
        if self.load_gate is not None:
            self.load_gate.wait()
        self.device = "fake-device"

    def synthesize_stream(self, text, ref_audio=None, ref_text=None, params=None):
        params = dict(params or {})
        self.calls.append({
            "method": "stream", "text": text, "ref_audio": ref_audio,
            "ref_text": ref_text, "params": params,
        })
        count = int(params.get("chunks", self.chunks))
        samples = int(params.get("samples", self.samples))
        produced = 0
        reason = "closed"
        try:
            if text == "BOOM":
                raise RuntimeError("fake engine failure")
            for index in range(count):
                if text == "MIDFAIL" and index == 1:
                    raise RuntimeError("fake failure mid-stream")
                if self.delay:
                    time.sleep(self.delay)
                if index and params.get("gap"):  # a pause between chunks
                    time.sleep(float(params["gap"]))
                yield Chunk(chunk_audio(index, samples), SAMPLE_RATE)
                produced += 1
            reason = "done"
        except RuntimeError:
            reason = "failed"
            raise
        finally:
            self.finished.append((text, reason, produced))
            self.finished_at = time.monotonic()
            self.finished_event.set()

    def synthesize(self, text, ref_audio=None, ref_text=None, params=None):
        chunks = list(self.synthesize_stream(text, ref_audio, ref_text, params))
        self.calls[-1]["method"] = "synthesize"
        audio = np.concatenate([c.audio for c in chunks])
        return Result(audio, SAMPLE_RATE, int(len(audio) / SAMPLE_RATE * 1000))

    def close(self) -> None:
        pass
