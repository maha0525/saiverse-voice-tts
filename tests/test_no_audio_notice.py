"""音声が 1 件も作られなかった吹き出しに、「音声は無い」を知らせる。

画面の音声ボタンは ``addon.json`` の ``show_when: "metadata_exists"`` 宣言で、
``audio_path`` が立つまで回転する待ち表示を出す。声にする文が一度も来なかった
吹き出しでは ``audio_path`` が永久に来ないので、待ち表示が残り続けていた。

``notify_no_audio`` は代わりに ``unavailable_keys`` (= この吹き出しではもう
立たない鍵の名前の配列) を立てて、画面に待つのをやめさせる。``audio_path`` に
偽の値を置かないのが肝心 — 再生成ボタンはその値の変化を完了の合図に使っている。
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

# Pack root を sys.path に追加
_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak import playback_worker as pw  # noqa: E402
from tools.speak import text_cleaner as _text_cleaner  # noqa: E402


class _MetadataSpy:
    """``saiverse.addon_metadata`` / ``addon_events`` の差し替え。

    実物は本体のデータベースを触るので、テストでは呼ばれた内容だけを覚える
    ダミーを ``sys.modules`` に置く (playback_worker は関数の中で遅延 import
    するため、この差し替えが効く)。
    """

    def __init__(self) -> None:
        self.metadata: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> "_MetadataSpy":
        meta_mod = types.ModuleType("saiverse.addon_metadata")
        meta_mod.set_metadata = lambda **kw: self.metadata.append(kw)  # type: ignore[attr-defined]
        event_mod = types.ModuleType("saiverse.addon_events")
        event_mod.emit_addon_event = lambda **kw: self.events.append(kw)  # type: ignore[attr-defined]
        pkg = sys.modules.get("saiverse") or types.ModuleType("saiverse")
        for name, mod in (
            ("saiverse", pkg),
            ("saiverse.addon_metadata", meta_mod),
            ("saiverse.addon_events", event_mod),
        ):
            self._saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class TestNotifyNoAudio(unittest.TestCase):
    def test_it_declares_the_keys_that_will_never_arrive(self):
        with _MetadataSpy() as spy:
            pw.notify_no_audio("msg-1", "nothing to say", pulse_id="pulse-1")

        self.assertEqual(len(spy.metadata), 1)
        written = spy.metadata[0]
        self.assertEqual(written["message_id"], "msg-1")
        self.assertEqual(written["key"], "unavailable_keys")
        self.assertIn("audio_path", written["value"])
        # 再生成ボタンは audio_path の値の変化を完了の合図にしている。
        # 偽の値を置くとその合図が壊れるので、ここでは触らない。
        self.assertNotIn(
            "audio_path", [m["key"] for m in spy.metadata],
        )

    def test_the_event_carries_the_same_declaration(self):
        """再読込を待たずに待ち表示が解けるよう、SSE にも同じ中身を載せる。"""
        with _MetadataSpy() as spy:
            pw.notify_no_audio("msg-1", "nothing to say", pulse_id="pulse-1")

        self.assertEqual(len(spy.events), 1)
        event = spy.events[0]
        self.assertEqual(event["message_id"], "msg-1")
        self.assertEqual(event["event"], "audio_unavailable")
        self.assertIn("audio_path", event["data"]["unavailable_keys"])
        self.assertEqual(event["data"]["pulse_id"], "pulse-1")

    def test_a_message_without_an_id_is_a_no_op(self):
        with _MetadataSpy() as spy:
            pw.notify_no_audio(None, "nothing to say")

        self.assertEqual(spy.metadata, [])
        self.assertEqual(spy.events, [])

    def test_a_close_request_for_an_unknown_message_tells_the_ui(self):
        """本丸: 知らない message の閉じ依頼を、黙って捨てずに知らせる。"""
        worker = pw._TTSWorker.__new__(pw._TTSWorker)
        worker._message_states = {}
        job = pw._Job(
            job_id="j1", text="", persona_id="p1",
            message_id="msg-unknown", pulse_id="pulse-1",
            sub_seq=None, is_final=True,
        )

        with _MetadataSpy() as spy:
            worker._process(job)

        self.assertEqual(
            [m["key"] for m in spy.metadata], ["unavailable_keys"],
        )
        self.assertEqual(spy.events[0]["message_id"], "msg-unknown")


class TestAutoSpeakOff(unittest.TestCase):
    """自動発話を切っているペルソナも、締めの合図で同じ知らせを出す。"""

    # 本体の tool loader は speak パックを ``tools._loaded.speak`` として
    # sys.modules に登録する (speak_hook はその名前で import している)。
    # テストではパック内の実モジュールを同じ名前に置いて成立させる。
    _ALIASES = ("tools._loaded", "tools._loaded.speak",
                "tools._loaded.speak.playback_worker",
                "tools._loaded.speak.text_cleaner")

    def setUp(self):
        self._saved = {n: sys.modules.get(n) for n in self._ALIASES}
        sys.modules["tools._loaded"] = types.ModuleType("tools._loaded")
        sys.modules["tools._loaded.speak"] = types.ModuleType("tools._loaded.speak")
        sys.modules["tools._loaded.speak.playback_worker"] = pw
        sys.modules["tools._loaded.speak.text_cleaner"] = _text_cleaner

    def tearDown(self):
        for name, mod in self._saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def test_the_final_signal_tells_the_ui_there_is_no_audio(self):
        import speak_hook

        original = speak_hook.get_effective_params
        speak_hook.get_effective_params = MagicMock(
            return_value={"_enabled": True, "auto_speak": False},
        )
        try:
            with _MetadataSpy() as spy:
                speak_hook.on_persona_speak(
                    "p1", "", "msg-1", is_final=True, pulse_id="pulse-1",
                )
            self.assertEqual(
                [m["key"] for m in spy.metadata], ["unavailable_keys"],
            )

            # 途中の sub-speak では何も書かない (締めの 1 回だけ)
            with _MetadataSpy() as spy2:
                speak_hook.on_persona_speak(
                    "p1", "こんにちは", "msg-1", is_final=False, sub_seq=1,
                )
            self.assertEqual(spy2.metadata, [])
        finally:
            speak_hook.get_effective_params = original


if __name__ == "__main__":
    unittest.main()
