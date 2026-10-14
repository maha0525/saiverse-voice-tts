"""仮想身体 (Godot Vessel アドオン) へ声を一切れずつ渡す経路。

設計: SAIVerse 本体 docs/intent/virtual_embodiment_godot.md の
「声は voice-tts から一切れずつ受け取る」。

- 仮想身体が引き受けた発話 (``register_voice_claim`` の判定が真) は、
  ``speak_hook`` が通常の合成キューへ入れない。判定は message_id ごとに
  最初の断片で一度だけ決まる。
- ``synthesize_piece`` は既存のワーカースレッドで一切れだけ合成し、
  WAV と秒数を返す。整形 (``clean_text_for_tts``) と auto_speak は通常の
  合成と同じに効く。
- ``finish_message`` は切れ端を依頼順につないで保存し、自動再生されない
  ``audio_completed`` で吹き出しに声を付ける。一切れでも欠けたら
  ``notify_no_audio``。
"""
from __future__ import annotations

import io
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak import playback_worker as pw  # noqa: E402
from tools.speak import pronunciation_dict  # noqa: E402
from tools.speak import text_cleaner as _text_cleaner  # noqa: E402
from tools.speak.engine.base import SynthesisResult, TTSEngine  # noqa: E402

SAMPLE_RATE = 24000
_AMPLITUDE = {"一つ目": 0.25, "二つ目": 0.5}


class _TextEngine(TTSEngine):
    """文ごとに長さと振幅が決まる音声を返すエンジン (モデルも GPU も使わない)。

    長さは ``len(text) * 120`` サンプル、振幅は ``_AMPLITUDE`` の値 (既定 0.1)。
    呼ばれたスレッドの名前を記録し、``gate`` が設定されていれば合成の前に待つ。
    """

    name = "gpt_sovits"
    supports_streaming = True

    def __init__(self) -> None:
        super().__init__({})
        self.texts: List[str] = []
        self.threads: List[str] = []
        self.gate: Optional[threading.Event] = None
        self.entered = threading.Event()

    def synthesize(self, text, ref_audio=None, ref_text=None, params=None) -> SynthesisResult:
        self.texts.append(text)
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        audio = np.full(len(text) * 120, _AMPLITUDE.get(text, 0.1), dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=SAMPLE_RATE,
                               duration_ms=int(len(audio) / SAMPLE_RATE * 1000))


class _VesselTestBase(unittest.TestCase):
    """本体のモジュール (設定・メタデータ・イベント) を差し替え、本物のワーカーを動かす。"""

    _ALIASES = ("tools._loaded", "tools._loaded.speak",
                "tools._loaded.speak.playback_worker",
                "tools._loaded.speak.text_cleaner")

    def setUp(self) -> None:
        self.params: Dict[str, Any] = {
            "_enabled": True, "auto_speak": True,
            "streaming": True, "server_side_playback": False,
        }
        self.metadata: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        addon_config = types.ModuleType("saiverse.addon_config")
        addon_config.get_params = lambda *a, **k: dict(self.params)  # type: ignore[attr-defined]
        meta_mod = types.ModuleType("saiverse.addon_metadata")
        meta_mod.set_metadata = lambda **kw: self.metadata.append(kw)  # type: ignore[attr-defined]
        event_mod = types.ModuleType("saiverse.addon_events")
        event_mod.emit_addon_event = lambda **kw: self.events.append(kw)  # type: ignore[attr-defined]
        modules = {
            "saiverse": sys.modules.get("saiverse") or types.ModuleType("saiverse"),
            "saiverse.addon_config": addon_config,
            "saiverse.addon_metadata": meta_mod,
            "saiverse.addon_events": event_mod,
            # 本体の tool loader は speak パックを tools._loaded.speak として登録する
            "tools._loaded": types.ModuleType("tools._loaded"),
            "tools._loaded.speak": types.ModuleType("tools._loaded.speak"),
            "tools._loaded.speak.playback_worker": pw,
            "tools._loaded.speak.text_cleaner": _text_cleaner,
        }
        self.stack.enter_context(patch.dict(sys.modules, modules))

        self.engine = _TextEngine()
        self.worker = pw._TTSWorker()
        self.worker._config_loaded = True
        self.worker._config = {"engines": {"gpt_sovits": {}}}
        self.stack.callback(self.worker.shutdown)
        self.stack.enter_context(patch.object(pw, "_worker", self.worker))
        self.stack.enter_context(patch.object(pw, "_out_dir", return_value=Path(self.tmp.name)))
        self.stack.enter_context(patch.object(pw, "get_profile", return_value={
            "engine": "gpt_sovits", "ref_audio": None, "ref_text": None, "params": {},
        }))
        self.stack.enter_context(
            patch.object(pronunciation_dict, "apply", side_effect=lambda t, **k: t))
        self.stack.enter_context(patch.object(self.worker, "_get_engine", return_value=self.engine))

        self._reset_vessel_state()
        self.addCleanup(self._reset_vessel_state)

    @staticmethod
    def _reset_vessel_state() -> None:
        with pw._vessel_lock:
            pw._voice_claims.clear()
            pw._vessel_messages.clear()
            pw._claim_decisions.clear()
            pw._finished_messages.clear()

    def event_names(self) -> List[str]:
        return [event["event"] for event in self.events]

    def metadata_keys(self) -> List[str]:
        return [row["key"] for row in self.metadata]


class SpeakHookClaimTests(_VesselTestBase):
    def setUp(self) -> None:
        super().setUp()
        import speak_hook
        self.speak_hook = speak_hook
        self.enqueue = self.stack.enter_context(
            patch.object(speak_hook, "enqueue_tts", return_value="JOB"))
        self.claimed = True
        self.claim_calls: List[tuple] = []

        def claim(persona_id: str, building_id: str) -> bool:
            self.claim_calls.append((persona_id, building_id))
            return self.claimed

        self.claim = claim
        pw.register_voice_claim(claim)

    def speak(self, text: str, message_id: str = "msg-1", **kwargs: Any) -> None:
        self.speak_hook.on_persona_speak(
            "p1", text, message_id, building_id="room", pulse_id="pulse-1", **kwargs)

    def test_claimed_message_is_not_enqueued_and_the_decision_is_kept(self):
        self.speak("一文目。", sub_seq=1, is_final=False)
        # 途中で Vessel が切れても、同じ発話の続きは同じ扱いのまま
        self.claimed = False
        self.speak("二文目。", sub_seq=2, is_final=False)
        self.speak("", is_final=True)

        self.enqueue.assert_not_called()
        self.assertEqual(self.claim_calls, [("p1", "room")])
        # 吹き出しの結末は finish_message が持つ。ここでは何も知らせない
        self.assertEqual(self.metadata, [])
        self.assertIn("msg-1", pw._vessel_messages)

        # 別の発話は改めて判定される
        self.speak("別の発話。", message_id="msg-2")
        self.enqueue.assert_called_once()
        self.assertEqual(self.enqueue.call_args.args[0], "別の発話。")

    def test_unclaimed_message_takes_the_normal_path(self):
        self.claimed = False
        self.speak("**こんにちは**", sub_seq=1, is_final=False)
        self.enqueue.assert_called_once_with(
            "こんにちは", "p1", message_id="msg-1", pulse_id="pulse-1",
            sub_seq=1, is_final=False,
        )
        self.assertNotIn("msg-1", pw._vessel_messages)

    def test_without_any_claim_function_nothing_changes(self):
        pw.unregister_voice_claim(self.claim)
        self.speak("こんにちは")
        self.enqueue.assert_called_once()

    def test_auto_speak_off_keeps_its_own_notice_before_any_claim(self):
        self.params["auto_speak"] = False
        self.speak("", is_final=True)
        self.enqueue.assert_not_called()
        self.assertEqual(self.claim_calls, [])
        self.assertEqual(self.metadata_keys(), ["unavailable_keys"])


class ClaimRegistryTests(_VesselTestBase):
    def test_registration_is_idempotent_and_unregister_removes_it(self):
        def claim(_persona: str, _building: str) -> bool:
            return True

        pw.register_voice_claim(claim)
        pw.register_voice_claim(claim)
        self.assertEqual(pw._voice_claims, [claim])
        self.assertTrue(pw._is_voice_claimed("p1", "room"))
        pw.unregister_voice_claim(claim)
        pw.unregister_voice_claim(claim)
        self.assertEqual(pw._voice_claims, [])
        self.assertFalse(pw._is_voice_claimed("p1", "room"))

    def test_a_failing_claim_counts_as_false(self):
        def broken(_persona: str, _building: str) -> bool:
            raise RuntimeError("boom")

        pw.register_voice_claim(broken)
        with self.assertLogs(pw.LOGGER, level="ERROR"):
            self.assertFalse(pw._is_voice_claimed("p1", "room"))


class SynthesizePieceTests(_VesselTestBase):
    def test_returns_a_wav_of_the_cleaned_text_made_on_the_worker_thread(self):
        audio, duration = pw.synthesize_piece(
            "p1", "**一つ目** https://example.com/x", message_id="msg-1", timeout=5)

        self.assertEqual(self.engine.texts, ["一つ目"])
        self.assertEqual(self.engine.threads, ["voice-tts-worker"])
        with wave.open(io.BytesIO(audio), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), SAMPLE_RATE)
            self.assertEqual(wf.getnframes(), 3 * 120)
            first = np.frombuffer(wf.readframes(1), dtype=np.int16)[0]
        self.assertAlmostEqual(duration, 3 * 120 / SAMPLE_RATE)
        self.assertEqual(first, int(0.25 * 32767))
        # 一切れの合成では再生も保存も画面への通知もしない
        self.assertEqual(self.events, [])
        self.assertEqual(self.metadata, [])
        self.assertEqual(list(Path(self.tmp.name).glob("*.wav")), [])

    def test_auto_speak_off_or_disabled_addon_makes_no_voice(self):
        for change in ({"auto_speak": False}, {"_enabled": False}):
            with self.subTest(change=change):
                saved = dict(self.params)
                self.params.update(change)
                with self.assertRaises(pw.VoiceUnavailable):
                    pw.synthesize_piece("p1", "一つ目", message_id="msg-off", timeout=5)
                self.params = saved
        self.assertEqual(self.engine.texts, [])
        pw.finish_message("msg-off", complete=True)
        self.assertEqual(self.metadata_keys(), ["unavailable_keys"])
        self.assertNotIn("audio_completed", self.event_names())

    def test_no_voice_profile_makes_no_voice(self):
        with patch.object(pw, "get_profile", return_value=None):
            with self.assertRaises(pw.VoiceUnavailable):
                pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)

    def test_timeout_raises_and_breaks_the_message(self):
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        with self.assertRaises(TimeoutError):
            pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=0.05)
        # 待ち行列にいる間に時間切れになった切れ端は合成されない
        with self.assertRaises(TimeoutError):
            pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=0.05)
        self.engine.gate.set()
        deadline = time.monotonic() + 5
        while self.worker._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.engine.texts, ["一つ目"])
        self.assertTrue(pw._vessel_messages["msg-1"].broken)

        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.metadata_keys(), ["unavailable_keys"])
        self.assertNotIn("audio_completed", self.event_names())

    def test_normal_jobs_and_pieces_share_one_worker_in_order(self):
        """通常の合成とエンジンを同時に使わない (同じワーカーの FIFO に並ぶ)。"""
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        self.params["streaming"] = False
        self.worker.enqueue("通常の発話", "p1", message_id="msg-normal")
        self.assertTrue(self.engine.entered.wait(5))
        result: Dict[str, Any] = {}

        def piece() -> None:
            result["value"] = pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)

        thread = threading.Thread(target=piece)
        thread.start()
        time.sleep(0.1)
        self.assertEqual(self.engine.texts, ["通常の発話"])  # 先の合成が終わるまで待つ
        self.engine.gate.set()
        thread.join(5)
        self.assertEqual(self.engine.texts, ["通常の発話", "一つ目"])
        self.assertEqual(set(self.engine.threads), {"voice-tts-worker"})
        self.assertIn("value", result)


class FinishMessageTests(_VesselTestBase):
    def claim(self, message_id: str = "msg-1") -> None:
        pw.register_voice_claim(lambda _p, _b: True)
        self.assertTrue(pw.claim_message_voice("p1", "room", message_id, pulse_id="pulse-1"))

    def test_complete_message_gets_the_joined_voice_without_auto_play(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)

        self.assertEqual(self.event_names(), ["audio_completed"])
        self.assertNotIn("audio_ready", self.event_names())
        self.assertEqual(self.events[0]["data"]["pulse_id"], "pulse-1")
        meta = {row["key"]: row["value"] for row in self.metadata}
        self.assertTrue(meta["audio_path"].startswith(
            "/api/addon/saiverse-voice-tts/audio/msg-1?v="))
        saved = Path(meta["audio_file"])
        self.assertEqual(saved.parent, Path(self.tmp.name))
        with wave.open(str(saved), "rb") as wf:
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.assertEqual(len(frames), 2 * 3 * 120)
        self.assertEqual(frames[0], int(0.25 * 32767))   # 一つ目が先
        self.assertEqual(frames[-1], int(0.5 * 32767))   # 二つ目が後
        self.assertNotIn("msg-1", pw._vessel_messages)

    def test_a_piece_arriving_after_finish_does_not_reopen_the_message(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        wav, _ = pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertNotIn("msg-1", pw._vessel_messages)
        self.assertEqual(self.event_names(), ["audio_completed"])

    def test_a_piece_with_nothing_to_voice_does_not_cost_the_bubble_voice(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        with self.assertRaises(pw.NothingToVoice):
            pw.synthesize_piece("p1", "https://example.com/x", message_id="msg-1", timeout=5)
        pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"])

    def test_incomplete_or_broken_message_gets_no_voice(self):
        self.claim("msg-1")
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=False)

        self.claim("msg-2")
        pw.synthesize_piece("p1", "一つ目", message_id="msg-2", timeout=5)
        self.params["auto_speak"] = False
        with self.assertRaises(pw.VoiceUnavailable):
            pw.synthesize_piece("p1", "二つ目", message_id="msg-2", timeout=5)
        pw.finish_message("msg-2", complete=True)

        self.assertEqual(self.event_names(), ["audio_unavailable", "audio_unavailable"])
        self.assertNotIn("audio_path", self.metadata_keys())
        self.assertEqual(list(Path(self.tmp.name).glob("*.wav")), [])

    def test_unknown_message_is_told_there_is_no_voice(self):
        pw.finish_message("msg-unknown", complete=True, pulse_id="pulse-9")
        pw.finish_message("msg-unknown", complete=True)
        self.assertEqual(self.event_names(), ["audio_unavailable", "audio_unavailable"])
        self.assertEqual(self.events[0]["data"]["pulse_id"], "pulse-9")

    def test_message_the_hook_did_not_claim_is_left_to_the_normal_path(self):
        pw.register_voice_claim(lambda _p, _b: False)
        self.assertFalse(pw.claim_message_voice("p1", "room", "msg-1"))
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.events, [])
        self.assertEqual(self.metadata, [])


class ExpiryTests(_VesselTestBase):
    def test_a_claimed_message_without_finish_expires_with_a_notice(self):
        pw.register_voice_claim(lambda _p, _b: True)
        pw.claim_message_voice("p1", "room", "msg-old", pulse_id="pulse-old")
        pw._vessel_messages["msg-old"].touched_at -= pw._VESSEL_MESSAGE_TTL_SEC + 1

        pw.claim_message_voice("p1", "room", "msg-new")

        self.assertNotIn("msg-old", pw._vessel_messages)
        self.assertIn("msg-new", pw._vessel_messages)
        self.assertEqual(self.event_names(), ["audio_unavailable"])
        self.assertEqual(self.events[0]["message_id"], "msg-old")
        self.assertEqual(self.events[0]["data"]["pulse_id"], "pulse-old")

    def test_the_store_is_bounded(self):
        pw.register_voice_claim(lambda _p, _b: True)
        with patch.object(pw, "_VESSEL_MESSAGE_MAX", 2):
            for index in range(4):
                pw.claim_message_voice("p1", "room", f"msg-{index}")
            self.assertEqual(list(pw._vessel_messages), ["msg-2", "msg-3"])
        self.assertEqual([e["message_id"] for e in self.events], ["msg-0", "msg-1"])
        with patch.object(pw, "_CLAIM_DECISION_MAX", 3):
            for index in range(5):
                pw.claim_message_voice("p1", "room", f"other-{index}")
            self.assertLessEqual(len(pw._claim_decisions), 3)


if __name__ == "__main__":
    unittest.main()
