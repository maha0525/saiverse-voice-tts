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
- message_id ごとの記録は ``claim_message_voice`` だけが作り、state が
  "unclaimed" / "open" / "closed" と移る。吹き出しへの結末の知らせは
  "open" から "closed" へ移る一回だけ (締め・期限切れ・上限での追い出し)。
"""
from __future__ import annotations

import importlib
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
from tools.speak import pronunciation_dict  # noqa: E402,F401
from tools.speak import text_cleaner as _text_cleaner  # noqa: E402
from tools.speak.engine.base import SynthesisResult, TTSEngine  # noqa: E402

SAMPLE_RATE = 24000


def _live_pronunciation_dict():
    """``_apply_pronunciation`` が呼ぶ時点の pronunciation_dict。

    他のテストが ``tools.speak`` を読み込み直すと、このファイルが import した
    モジュールと ``from . import pronunciation_dict`` が解決するモジュールが
    別物になるので、差し替えは呼ぶ時点のものに当てる。
    """
    return importlib.import_module(pw.__package__ + ".pronunciation_dict")
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


def _always_claim(_persona: str, _building: str) -> bool:
    return True


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
            patch.object(_live_pronunciation_dict(), "apply", side_effect=lambda t, **k: t))
        self.stack.enter_context(patch.object(self.worker, "_get_engine", return_value=self.engine))

        self._reset_vessel_state()
        self.addCleanup(self._reset_vessel_state)

    @staticmethod
    def _reset_vessel_state() -> None:
        with pw._vessel_lock:
            pw._voice_claims.clear()
            pw._vessel_messages.clear()
            sweeper, pw._vessel_sweeper = pw._vessel_sweeper, None
        if sweeper is not None:
            sweeper.stop.set()
            sweeper.thread.join(5)

    def state_of(self, message_id: str) -> Optional[str]:
        with pw._vessel_lock:
            entry = pw._vessel_messages.get(message_id)
            return entry.state if entry is not None else None

    def age(self, *message_ids: str) -> None:
        """記録を ``_VESSEL_MESSAGE_TTL_SEC`` より前に触られたことにする。"""
        with pw._vessel_lock:
            for message_id in message_ids:
                pw._vessel_messages[message_id].touched_at -= pw._VESSEL_MESSAGE_TTL_SEC + 1

    def claim(self, message_id: str = "msg-1") -> None:
        """Godot アドオンと同じく、切れ端の前に ``claim_message_voice`` で引き受ける。"""
        pw.register_voice_claim(_always_claim)
        self.assertTrue(pw.claim_message_voice("p1", "room", message_id, pulse_id="pulse-1"))

    def wait_until(self, predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

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

        self.claim_fn = claim
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
        self.assertEqual(self.state_of("msg-1"), "open")

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
        self.assertEqual(self.state_of("msg-1"), "unclaimed")

    def test_without_any_claim_function_nothing_changes(self):
        pw.unregister_voice_claim(self.claim_fn)
        self.speak("こんにちは")
        self.enqueue.assert_called_once()

    def test_auto_speak_off_unclaimed_message_keeps_its_own_notice(self):
        self.claimed = False
        self.params["auto_speak"] = False
        self.speak("", is_final=True)
        self.enqueue.assert_not_called()
        self.assertEqual(self.claim_calls, [("p1", "room")])
        self.assertEqual(self.metadata_keys(), ["unavailable_keys"])

    def test_auto_speak_off_claimed_single_final_is_told_once_in_either_hook_order(self):
        """自動発話 OFF の単発の発話で、どちらのフックが先に走っても「音声なし」は一度だけ。"""
        self.params["auto_speak"] = False
        for order, message_id in ((("hook", "vessel"), "msg-a"), (("vessel", "hook"), "msg-b")):
            for who in order:
                if who == "hook":
                    self.speak("内緒。", message_id=message_id, is_final=True)
                else:
                    self.assertTrue(pw.claim_message_voice("p1", "room", message_id))
                    with self.assertRaises(pw.VoiceUnavailable):
                        pw.synthesize_piece("p1", "内緒。", message_id=message_id, timeout=5)
                    pw.finish_message(message_id, complete=True)
        self.enqueue.assert_not_called()
        self.assertEqual([e["message_id"] for e in self.events], ["msg-a", "msg-b"])
        self.assertEqual(self.event_names(), ["audio_unavailable", "audio_unavailable"])


class ClaimRegistryTests(_VesselTestBase):
    def test_registration_is_idempotent_and_unregister_removes_it(self):
        def claim(_persona: str, _building: str) -> bool:
            return True

        pw.register_voice_claim(claim)
        pw.register_voice_claim(claim)
        self.assertEqual(pw._voice_claims, [(claim, None)])
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

    def test_registering_with_holds_twice_stays_one_entry(self):
        def claim(_persona: str, _building: str) -> bool:
            return True

        def holds(_message_id: str) -> bool:
            return True

        pw.register_voice_claim(claim, holds)
        pw.register_voice_claim(claim, holds)
        self.assertEqual(pw._voice_claims, [(claim, holds)])
        self.assertTrue(pw._vessel_held_by_addon("msg-1"))
        pw.unregister_voice_claim(claim)
        self.assertEqual(pw._voice_claims, [])
        self.assertFalse(pw._vessel_held_by_addon("msg-1"))


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
        self.claim("msg-off")
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
        self.claim()
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        with self.assertRaises(TimeoutError):
            pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=0.05)
        # 待ち行列にいる間に時間切れになった切れ端は合成されない
        with self.assertRaises(TimeoutError):
            pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=0.05)
        self.engine.gate.set()
        self.assertTrue(self.wait_until(lambda: not self.worker._queue.unfinished_tasks))
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

    def test_a_queued_piece_is_not_synthesized_after_auto_speak_is_turned_off(self):
        """受付の後、合成を始める前に auto_speak が切られた切れ端は合成しない。"""
        self.claim()
        self.engine.gate = threading.Event()
        self.addCleanup(self.engine.gate.set)
        self.params["streaming"] = False
        self.worker.enqueue("通常の発話", "p1", message_id="msg-normal")
        self.assertTrue(self.engine.entered.wait(5))
        result: Dict[str, Any] = {}

        def piece() -> None:
            try:
                pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
            except Exception as exc:  # noqa: BLE001 - 送出された例外を確かめる
                result["error"] = exc

        thread = threading.Thread(target=piece)
        thread.start()
        # 切れ端がワーカーの待ち行列に並ぶまで待ってから切る
        self.assertTrue(self.wait_until(lambda: self.worker._queue.qsize() >= 1))
        self.params["auto_speak"] = False
        self.engine.gate.set()
        thread.join(5)

        self.assertIsInstance(result.get("error"), pw.VoiceUnavailable)
        self.assertEqual(self.engine.texts, ["通常の発話"])
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(
            [e["event"] for e in self.events if e["message_id"] == "msg-1"],
            ["audio_unavailable"],
        )

    def test_a_piece_of_an_unclaimed_or_unknown_message_is_not_kept(self):
        pw.register_voice_claim(lambda _p, _b: False)
        self.assertFalse(pw.claim_message_voice("p1", "room", "msg-normal"))
        for message_id in ("msg-normal", "msg-unknown"):
            wav, _ = pw.synthesize_piece("p1", "一つ目", message_id=message_id, timeout=5)
            self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(pw._vessel_messages["msg-normal"].pieces, [])
        self.assertNotIn("msg-unknown", pw._vessel_messages)


class FinishMessageTests(_VesselTestBase):
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
        self.assertEqual(self.state_of("msg-1"), "closed")
        self.assertEqual(pw._vessel_messages["msg-1"].pieces, [])

    def test_a_repeated_finish_tells_the_bubble_only_once(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        pw.finish_message("msg-1", complete=True)
        pw.finish_message("msg-1", complete=False)
        self.assertEqual(self.event_names(), ["audio_completed"])
        self.assertEqual(self.metadata_keys().count("unavailable_keys"), 0)

        self.claim("msg-2")
        pw.finish_message("msg-2", complete=False)
        pw.finish_message("msg-2", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed", "audio_unavailable"])

    def test_a_piece_arriving_after_finish_does_not_reopen_the_message(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        wav, _ = pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(self.state_of("msg-1"), "closed")
        self.assertEqual(pw._vessel_messages["msg-1"].pieces, [])
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"])
        # 締めの後の断片でも判定は変わらない (通常の合成へ流さない)
        self.assertTrue(pw.claim_message_voice("p1", "room", "msg-1"))

    def test_a_piece_with_nothing_to_voice_does_not_cost_the_bubble_voice(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        with self.assertRaises(pw.NothingToVoice):
            pw.synthesize_piece("p1", "https://example.com/x", message_id="msg-1", timeout=5)
        pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"])

    def test_a_piece_silenced_by_the_pronunciation_dict_is_left_out_of_the_voice(self):
        """読み方の辞書で読み上げる文字が消える切れ端も、欠けにせず飛ばしてつなぐ。"""
        self.stack.enter_context(patch.object(
            _live_pronunciation_dict(), "apply",
            side_effect=lambda t, **k: "" if t == "消える" else t))
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        with self.assertRaises(pw.NothingToVoice):
            pw.synthesize_piece("p1", "消える", message_id="msg-1", timeout=5)
        pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)

        self.assertEqual(self.engine.texts, ["一つ目", "二つ目"])
        self.assertEqual(self.event_names(), ["audio_completed"])
        meta = {row["key"]: row["value"] for row in self.metadata}
        with wave.open(str(meta["audio_file"]), "rb") as wf:
            frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        self.assertEqual(len(frames), 2 * 3 * 120)
        self.assertEqual(frames[0], int(0.25 * 32767))   # 一つ目
        self.assertEqual(frames[-1], int(0.5 * 32767))   # 二つ目

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

        self.claim("msg-3")
        pw.finish_message("msg-3", complete=True)  # 切れ端が一つも無い

        self.assertEqual(self.event_names(), ["audio_unavailable"] * 3)
        self.assertNotIn("audio_path", self.metadata_keys())
        self.assertEqual(list(Path(self.tmp.name).glob("*.wav")), [])

    def test_unknown_message_is_left_alone(self):
        """claim_message_voice を経ていない発話の吹き出しには何も知らせない。"""
        pw.finish_message("msg-unknown", complete=True, pulse_id="pulse-9")
        pw.finish_message("msg-unknown", complete=False)
        self.assertEqual(self.events, [])
        self.assertEqual(self.metadata, [])
        self.assertNotIn("msg-unknown", pw._vessel_messages)

    def test_message_the_hook_did_not_claim_is_left_to_the_normal_path(self):
        pw.register_voice_claim(lambda _p, _b: False)
        self.assertFalse(pw.claim_message_voice("p1", "room", "msg-1"))
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.events, [])
        self.assertEqual(self.metadata, [])
        self.assertEqual(self.state_of("msg-1"), "unclaimed")


class ClaimDecisionTests(_VesselTestBase):
    def test_concurrent_first_fragments_run_the_claim_function_once(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        entered = threading.Event()
        calls: List[str] = []
        answers = iter([True, False])

        def claim(_persona: str, _building: str) -> bool:
            calls.append(threading.current_thread().name)
            entered.set()
            gate.wait(5)
            return next(answers)

        pw.register_voice_claim(claim)
        results: Dict[str, bool] = {}

        def first_fragment(name: str) -> None:
            results[name] = pw.claim_message_voice("p1", "room", "msg-1")

        first = threading.Thread(target=first_fragment, args=("first",), name="first")
        second = threading.Thread(target=first_fragment, args=("second",), name="second")
        first.start()
        self.assertTrue(entered.wait(5))
        second.start()
        time.sleep(0.1)  # second が判定の順番待ちに入る
        gate.set()
        first.join(5)
        second.join(5)

        self.assertEqual(calls, ["first"])
        self.assertEqual(results, {"first": True, "second": True})
        self.assertEqual(self.state_of("msg-1"), "open")

    def test_the_bound_never_drops_a_decision_that_traffic_can_still_reach(self):
        """件数の目安を超えても、"open" と、終わってから 1 日たっていない記録は捨てない。

        捨てるのは 1 日以上前に終わった記録だけ。進行中の判定がやり直されたり、
        閉じた発話が新しい発話として開き直されたりしない。
        """
        claimed = {"value": True}
        calls: List[str] = []

        def claim(_persona: str, _building: str) -> bool:
            calls.append("call")
            return claimed["value"]

        pw.register_voice_claim(claim)
        with patch.object(pw, "_VESSEL_MESSAGE_MAX", 2):
            self.assertTrue(pw.claim_message_voice("p1", "room", "live"))
            self.assertTrue(pw.claim_message_voice("p1", "room", "done"))
            pw.finish_message("done", complete=False)
            self.assertTrue(pw.claim_message_voice("p1", "room", "ancient"))
            pw.finish_message("ancient", complete=False)
            with pw._vessel_lock:
                pw._vessel_messages["ancient"].touched_at -= pw._VESSEL_RETAIN_TERMINAL_SEC + 1
            claimed["value"] = False
            for message_id in ("new-0", "new-1", "new-2"):
                self.assertFalse(pw.claim_message_voice("p1", "room", message_id))
            # 1 日前に終わった "ancient" だけが捨てられ、目安を超えても他は残る
            self.assertNotIn("ancient", pw._vessel_messages)
            self.assertEqual(set(pw._vessel_messages), {"live", "done", "new-0", "new-1", "new-2"})
            calls_before = len(calls)

            # 判定は変わらず、関数も呼び直さない
            self.assertTrue(pw.claim_message_voice("p1", "room", "live"))
            self.assertTrue(pw.claim_message_voice("p1", "room", "done"))
            self.assertFalse(pw.claim_message_voice("p1", "room", "new-0"))
            self.assertEqual(len(calls), calls_before)

            # 閉じた発話に遅れて来た切れ端と締めは何も変えない
            pw.synthesize_piece("p1", "一つ目", message_id="done", timeout=5)
            pw.finish_message("done", complete=True)
            self.assertEqual(self.state_of("done"), "closed")
        self.assertEqual([e["message_id"] for e in self.events], ["done", "ancient"])


class ExpiryTests(_VesselTestBase):
    def test_a_claimed_message_without_finish_expires_with_a_notice(self):
        pw.register_voice_claim(lambda _p, _b: True)
        pw.claim_message_voice("p1", "room", "msg-old", pulse_id="pulse-old")
        self.age("msg-old")

        pw.claim_message_voice("p1", "room", "msg-new")

        self.assertEqual(self.state_of("msg-old"), "closed")
        self.assertEqual(self.state_of("msg-new"), "open")
        self.assertEqual(self.event_names(), ["audio_unavailable"])
        self.assertEqual(self.events[0]["message_id"], "msg-old")
        self.assertEqual(self.events[0]["data"]["pulse_id"], "pulse-old")

    def test_the_sweeper_expires_an_open_message_without_further_calls(self):
        with patch.object(pw, "_VESSEL_MESSAGE_TTL_SEC", 0.05), \
                patch.object(pw, "_VESSEL_SWEEP_INTERVAL_SEC", 0.02):
            self.claim()
            self.assertTrue(self.wait_until(lambda: self.event_names() == ["audio_unavailable"]))
            self.assertEqual(self.events[0]["message_id"], "msg-1")
            self.assertEqual(self.state_of("msg-1"), "closed")
            # "open" の記録が無くなったら見回りは自分で抜ける
            self.assertTrue(self.wait_until(lambda: pw._vessel_sweeper is None))
            time.sleep(0.1)
            self.assertEqual(self.event_names(), ["audio_unavailable"])

            # 次に "open" になったら改めて起きる
            self.claim("msg-2")
            self.assertTrue(self.wait_until(lambda: len(self.events) == 2))
            self.assertEqual(self.events[1]["message_id"], "msg-2")

    def test_late_pieces_after_expiry_do_not_reopen_or_complete_the_message(self):
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        self.age("msg-1")
        pw._expire_vessel_messages()
        self.assertEqual(self.event_names(), ["audio_unavailable"])

        wav, _ = pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertEqual(self.state_of("msg-1"), "closed")
        self.assertEqual(pw._vessel_messages["msg-1"].pieces, [])
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_unavailable"])
        self.assertEqual(list(Path(self.tmp.name).glob("*.wav")), [])

    def test_a_message_the_addon_still_holds_survives_the_ttl_until_finished(self):
        held = {"msg-1"}
        pw.register_voice_claim(_always_claim, lambda message_id: message_id in held)
        self.assertTrue(pw.claim_message_voice("p1", "room", "msg-1", pulse_id="pulse-1"))
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        for _ in range(5):
            self.age("msg-1")
            pw._expire_vessel_messages()
        self.assertEqual(self.state_of("msg-1"), "open")
        self.assertEqual(self.events, [])

        pw.synthesize_piece("p1", "二つ目", message_id="msg-1", timeout=5)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"])

    def test_once_holds_answers_false_the_message_expires_once(self):
        held = {"msg-1"}
        pw.register_voice_claim(_always_claim, lambda message_id: message_id in held)
        self.assertTrue(pw.claim_message_voice("p1", "room", "msg-1"))
        self.age("msg-1")
        pw._expire_vessel_messages()
        self.assertEqual(self.state_of("msg-1"), "open")

        held.clear()
        self.age("msg-1")
        pw._expire_vessel_messages()
        pw._expire_vessel_messages()
        self.assertEqual(self.state_of("msg-1"), "closed")
        self.assertEqual(self.event_names(), ["audio_unavailable"])

    def test_a_failing_holds_counts_as_false(self):
        def broken(_message_id: str) -> bool:
            raise RuntimeError("boom")

        pw.register_voice_claim(_always_claim, broken)
        self.assertTrue(pw.claim_message_voice("p1", "room", "msg-1"))
        self.age("msg-1")
        with self.assertLogs(pw.LOGGER, level="ERROR"):
            pw._expire_vessel_messages()
        self.assertEqual(self.state_of("msg-1"), "closed")
        self.assertEqual(self.event_names(), ["audio_unavailable"])

    def test_open_messages_are_never_pushed_out_by_the_bound(self):
        pw.register_voice_claim(lambda _p, _b: True)
        with patch.object(pw, "_VESSEL_MESSAGE_MAX", 2):
            for index in range(4):
                pw.claim_message_voice("p1", "room", f"msg-{index}")
            self.assertEqual(list(pw._vessel_messages), [f"msg-{i}" for i in range(4)])
            for index in range(4):
                pw.synthesize_piece("p1", "一つ目", message_id=f"msg-{index}", timeout=5)
                pw.finish_message(f"msg-{index}", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"] * 4)

    def test_auto_speak_turned_off_mid_message_tells_the_bubble_once(self):
        """仮想身体が引き受けた発話では、自動発話 OFF の「音声なし」を speak_hook が重ねない。"""
        self.claim()
        pw.synthesize_piece("p1", "一つ目", message_id="msg-1", timeout=5)
        self.params["auto_speak"] = False
        import speak_hook

        speak_hook.on_persona_speak("p1", "", "msg-1", building_id="room", is_final=True)
        pw.finish_message("msg-1", complete=True)
        self.assertEqual(self.event_names(), ["audio_completed"])


if __name__ == "__main__":
    unittest.main()
