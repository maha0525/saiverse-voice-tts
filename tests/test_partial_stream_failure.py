"""A sentence whose stream fails after some audio was played is not synthesized again.

Re-synthesizing the whole sentence would play the already-played first half a
second time. This happens most easily with an external synthesis server whose
connection drops mid-sentence, but the rule is the same for every engine.
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import numpy as np
import soundfile as sf

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak import playback_worker as pw  # noqa: E402
from tools.speak import pronunciation_dict  # noqa: E402
from tools.speak.engine.base import SynthesisChunk, SynthesisResult, TTSEngine  # noqa: E402

SAMPLE_RATE = 24000


class _BreakingEngine(TTSEngine):
    """Streams ``good_chunks`` chunks, then fails (unless ``good_chunks`` is None)."""

    name = "gpt_sovits"
    supports_streaming = True

    def __init__(self, good_chunks: Optional[int]):
        super().__init__({})
        self.good_chunks = good_chunks
        self.whole_sentence_calls: List[str] = []

    def synthesize_stream(self, text, ref_audio=None, ref_text=None, params=None) -> Iterator[SynthesisChunk]:
        for _ in range(self.good_chunks or 0):
            yield SynthesisChunk(audio=np.full(240, 0.25, dtype=np.float32), sample_rate=SAMPLE_RATE)
        if self.good_chunks is not None:
            raise RuntimeError("connection dropped mid-sentence")

    def synthesize(self, text, ref_audio=None, ref_text=None, params=None) -> SynthesisResult:
        self.whole_sentence_calls.append(text)
        audio = np.full(480, 0.5, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=SAMPLE_RATE, duration_ms=20)


class PartialStreamFailureTests(unittest.TestCase):
    def _run(self, engine: _BreakingEngine, jobs: List[Dict[str, Any]]):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            worker = pw._TTSWorker()
            worker._config_loaded = True
            worker._config = {"engines": {"gpt_sovits": {}}}
            addon_config = types.ModuleType("saiverse.addon_config")
            addon_config.get_params = lambda *a, **k: {
                "engine": "gpt_sovits", "streaming": True, "server_side_playback": False,
            }
            stack.enter_context(patch.dict(sys.modules, {"saiverse.addon_config": addon_config}))
            stack.enter_context(patch.object(pw, "_worker", worker))
            stack.enter_context(patch.object(pw, "_out_dir", return_value=Path(tmp)))
            stack.enter_context(patch.object(pw, "get_profile", return_value={
                "engine": "gpt_sovits", "ref_audio": None, "ref_text": None, "params": {},
            }))
            stack.enter_context(patch.object(pronunciation_dict, "apply", side_effect=lambda t, **k: t))
            stack.enter_context(patch.object(worker, "_get_engine", return_value=engine))
            ready = stack.enter_context(patch.object(pw, "_notify_audio_ready"))
            stack.enter_context(patch.object(pw, "_notify_stream_ready"))
            no_audio = stack.enter_context(patch.object(pw, "notify_no_audio"))
            stream = stack.enter_context(patch.object(pw, "audio_stream"))
            for job in jobs:
                worker._process(pw._Job(persona_id="synthetic", message_id="msg", **job))
            wavs = [sf.read(str(p))[0] for p in sorted(Path(tmp).glob("*.wav"))]
            return worker, ready, no_audio, stream, wavs

    def test_failure_after_audio_keeps_the_played_part_and_does_not_resynthesize(self):
        engine = _BreakingEngine(good_chunks=2)
        worker, ready, no_audio, stream, wavs = self._run(
            engine, [{"text": "hello", "job_id": "j1"}],
        )
        self.assertEqual(engine.whole_sentence_calls, [])  # nothing is played twice
        self.assertEqual(stream.push_chunk.call_count, 2)
        # the bubble still gets the audio that was played
        self.assertEqual(len(wavs), 1)
        self.assertEqual(len(wavs[0]), 480)
        self.assertEqual(ready.call_args.kwargs.get("event_name"), "audio_completed")
        no_audio.assert_not_called()
        self.assertNotIn("msg", worker._message_states)

    def test_mid_message_failure_continues_with_the_next_sentence(self):
        engine = _BreakingEngine(good_chunks=1)
        worker, ready, no_audio, stream, wavs = self._run(engine, [
            {"text": "first", "job_id": "j1", "sub_seq": 1, "is_final": False},
        ])
        self.assertEqual(engine.whole_sentence_calls, [])
        # the stream for this bubble stays open for the next sentence
        stream.close_stream.assert_not_called()
        self.assertIn("msg", worker._message_states)

    def test_failure_before_any_audio_still_falls_back_to_whole_sentence(self):
        engine = _BreakingEngine(good_chunks=0)
        worker, ready, no_audio, stream, wavs = self._run(
            engine, [{"text": "hello", "job_id": "j1"}],
        )
        self.assertEqual(engine.whole_sentence_calls, ["hello"])
        self.assertEqual(len(wavs), 1)
        self.assertEqual(len(wavs[0]), 480)
        no_audio.assert_not_called()


if __name__ == "__main__":
    unittest.main()
