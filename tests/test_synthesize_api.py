"""Tests for direct voice test synthesis (playback_worker.synthesize_wav)
and POST /synthesize, POST /synthesize-batch in api_routes.py.
"""
from __future__ import annotations

import io
import json
import sys
import types
import unittest
import urllib.parse
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


def _make_dummy_wav(samples: int = 2400, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * samples)
    return buf.getvalue()


@dataclass
class _FakeResult:
    wav_bytes: bytes
    duration_sec: float
    sample_rate: int
    engine_name: str
    applied_text: str


class _FakeVoiceUnavailable(Exception):
    pass


class _FakeNothingToVoice(_FakeVoiceUnavailable):
    pass


def _make_app() -> FastAPI:
    if "api_routes" in sys.modules:
        del sys.modules["api_routes"]
    import api_routes  # type: ignore

    app = FastAPI()
    app.include_router(api_routes.router, prefix="/api/addon/saiverse-voice-tts")

    async def _dep() -> Any:
        return object()

    for route in api_routes.router.routes:
        deps = getattr(route, "dependant", None)
        if deps is None:
            continue
        for sub in list(deps.dependencies):
            app.dependency_overrides[sub.call] = _dep
    return app


class FilenameSanitizationTests(unittest.TestCase):
    def setUp(self):
        if "api_routes" in sys.modules:
            del sys.modules["api_routes"]
        import api_routes  # type: ignore

        self.api = api_routes

    def test_sanitize_blocks_traversal_and_reserved_names(self):
        self.assertEqual(
            self.api._sanitize_wav_filename("../secret/evil", "001"),
            "secret_evil.wav",
        )
        self.assertEqual(
            self.api._sanitize_wav_filename("a\\b:c*d?e", "001"),
            "a_b_c_d_e.wav",
        )
        self.assertEqual(
            self.api._sanitize_wav_filename("CON", "001"),
            "CON_.wav",
        )
        self.assertEqual(
            self.api._sanitize_wav_filename("greeting.wav", "001"),
            "greeting.wav",
        )
        self.assertEqual(
            self.api._sanitize_wav_filename("   ", "001"),
            "001.wav",
        )
        self.assertEqual(
            self.api._sanitize_wav_filename("おはよう", "001"),
            "おはよう.wav",
        )

    def test_unique_filename_dedupes_case_insensitively(self):
        used = {"manifest.json"}
        self.assertEqual(self.api._unique_wav_filename("Hello.wav", used), "Hello.wav")
        self.assertEqual(self.api._unique_wav_filename("hello.wav", used), "hello_2.wav")
        self.assertEqual(self.api._unique_wav_filename("hello.wav", used), "hello_3.wav")

    def test_sanitize_and_unique_bound_utf8_byte_length_for_multibyte_names(self):
        jp = self.api._sanitize_wav_filename("あ" * 100, "001")
        self.assertTrue(jp.endswith(".wav"))
        self.assertLessEqual(len(jp.encode("utf-8")), 204)
        self.assertEqual(jp, ("あ" * 66) + ".wav")

        emoji = self.api._sanitize_wav_filename("🎵" * 80, "001")
        self.assertLessEqual(len(emoji.encode("utf-8")), 204)
        self.assertEqual(emoji, ("🎵" * 50) + ".wav")

        used: set[str] = set()
        first = self.api._unique_wav_filename(jp, used)
        second = self.api._unique_wav_filename(jp, used)
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(second.encode("utf-8")), 204)


class SynthesizeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.fake_pw = types.ModuleType("tools._loaded.speak.playback_worker")
        self.fake_pw.VoiceUnavailable = _FakeVoiceUnavailable  # type: ignore[attr-defined]
        self.fake_pw.NothingToVoice = _FakeNothingToVoice  # type: ignore[attr-defined]
        self.mock_synth = MagicMock(
            return_value=_FakeResult(
                wav_bytes=_make_dummy_wav(),
                duration_sec=0.1,
                sample_rate=24000,
                engine_name="elevenlabs",
                applied_text="マハー、こんにちは",
            )
        )
        self.fake_pw.synthesize_wav = self.mock_synth  # type: ignore[attr-defined]
        self._prev_pw = sys.modules.get("tools._loaded.speak.playback_worker")
        sys.modules["tools._loaded.speak.playback_worker"] = self.fake_pw

        self.app = _make_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        if self._prev_pw is None:
            sys.modules.pop("tools._loaded.speak.playback_worker", None)
        else:
            sys.modules["tools._loaded.speak.playback_worker"] = self._prev_pw

    def test_single_synthesize_returns_wav_and_metadata_headers(self):
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={
                "persona_id": "air_city_a",
                "text": "まはー、こんにちは",
                "apply_pronunciation": True,
                "filename": "挨拶テスト",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers["content-type"], "audio/wav")
        self.assertTrue(res.content.startswith(b"RIFF"))
        self.assertEqual(res.headers["x-tts-engine"], "elevenlabs")
        self.assertEqual(res.headers["x-tts-sample-rate"], "24000")
        self.assertEqual(res.headers["x-tts-duration-sec"], "0.100")
        self.assertEqual(
            urllib.parse.unquote(res.headers["x-tts-applied-text"]),
            "マハー、こんにちは",
        )
        self.assertIn("filename*=UTF-8''", res.headers["content-disposition"])
        self.mock_synth.assert_called_once_with(
            persona_id="air_city_a",
            text="まはー、こんにちは",
            apply_pronunciation=True,
            clean_text=True,
            timeout=180.0,
        )

    def test_single_synthesize_json_format_preserves_full_long_applied_text(self):
        import base64

        long_text = "あ" * 650
        self.mock_synth.return_value = _FakeResult(
            wav_bytes=_make_dummy_wav(),
            duration_sec=1.234,
            sample_rate=24000,
            engine_name="elevenlabs",
            applied_text=long_text,
        )
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={
                "persona_id": "air_city_a",
                "text": long_text,
                "response_format": "json",
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["applied_text"], long_text)
        self.assertEqual(len(body["applied_text"]), 650)
        self.assertEqual(body["engine"], "elevenlabs")
        self.assertEqual(body["sample_rate"], 24000)
        self.assertTrue(base64.b64decode(body["audio_base64"]).startswith(b"RIFF"))

        # WAV モードではヘッダプレビューが切り詰められた旨を明示し、不正フォーマットは 400
        res_wav = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": long_text},
        )
        self.assertEqual(res_wav.status_code, 200)
        self.assertEqual(res_wav.headers["x-tts-applied-text-truncated"], "true")

        res_bad = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": "hi", "response_format": "xml"},
        )
        self.assertEqual(res_bad.status_code, 400)

    def test_single_synthesize_validation_and_error_mapping(self):
        res = self.client.post("/api/addon/saiverse-voice-tts/synthesize", json={"text": "hi"})
        self.assertEqual(res.status_code, 400)

        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": "   "},
        )
        self.assertEqual(res.status_code, 400)

        self.mock_synth.side_effect = _FakeNothingToVoice("empty after clean")
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": "https://example.com"},
        )
        self.assertEqual(res.status_code, 400)

        self.mock_synth.side_effect = TimeoutError("timed out")
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": "hello"},
        )
        self.assertEqual(res.status_code, 504)

        self.mock_synth.side_effect = RuntimeError("ElevenLabs 402 paid_plan_required")
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize",
            json={"persona_id": "air_city_a", "text": "hello"},
        )
        self.assertEqual(res.status_code, 502)
        self.assertIn("paid_plan_required", res.json()["detail"])

    def test_batch_synthesize_mixed_items_partial_failure_and_zip_manifest(self):
        calls: List[Dict[str, Any]] = []

        def _fake_synth(*, persona_id, text, apply_pronunciation, clean_text, timeout):
            calls.append({"persona_id": persona_id, "text": text})
            if text == "BOOM":
                raise RuntimeError("engine refused")
            return _FakeResult(
                wav_bytes=_make_dummy_wav(),
                duration_sec=0.25,
                sample_rate=24000,
                engine_name="openai_tts",
                applied_text=text,
            )

        self.mock_synth.side_effect = _fake_synth
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize-batch",
            json={
                "persona_id": "air_city_a",
                "items": [
                    "Hello, world!",
                    {"filename": "greet", "text": "おはよう！"},
                    {"filename": "greet", "text": "重複名テスト"},
                    {"filename": "../escape", "text": "パス抜け防止", "persona_id": "yui_city_a"},
                    "BOOM",
                ],
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.headers["content-type"], "application/zip")
        self.assertEqual(res.headers["x-tts-batch-total"], "5")
        self.assertEqual(res.headers["x-tts-batch-succeeded"], "4")
        self.assertEqual(res.headers["x-tts-batch-failed"], "1")

        # 4th item used overridden persona_id
        self.assertEqual(calls[3]["persona_id"], "yui_city_a")

        with zipfile.ZipFile(io.BytesIO(res.content), "r") as zf:
            names = set(zf.namelist())
            self.assertEqual(
                names,
                {"001.wav", "greet.wav", "greet_2.wav", "escape.wav", "manifest.json"},
            )
            self.assertTrue(zf.read("001.wav").startswith(b"RIFF"))
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest["total"], 5)
            self.assertEqual(manifest["succeeded"], 4)
            self.assertEqual(manifest["failed"], 1)
            self.assertEqual(manifest["items"][4]["status"], "error")
            self.assertIn("engine refused", manifest["items"][4]["error"])

    def test_batch_all_failed_returns_error(self):
        self.mock_synth.side_effect = _FakeVoiceUnavailable("no voice profile")
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize-batch",
            json={"persona_id": "air_city_a", "items": ["a", "b"]},
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("no voice profile", res.json()["detail"])

    def test_batch_rejects_empty_and_over_limit(self):
        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize-batch",
            json={"persona_id": "air_city_a", "items": []},
        )
        self.assertEqual(res.status_code, 400)

        res = self.client.post(
            "/api/addon/saiverse-voice-tts/synthesize-batch",
            json={"persona_id": "air_city_a", "items": ["x"] * 201},
        )
        self.assertEqual(res.status_code, 400)


class PlaybackWorkerDirectSynthesizeTests(unittest.TestCase):
    """Real playback_worker.synthesize_wav test with a fake engine."""

    def test_synthesize_wav_works_when_auto_speak_is_off_and_toggles_pronunciation(self):
        from tools.speak import playback_worker as pw
        from tools.speak.engine.base import SynthesisResult

        captured_texts: List[str] = []

        class _DummyEngine:
            supports_streaming = False

            def synthesize(self, text, ref_audio=None, ref_text=None, params=None):
                captured_texts.append(text)
                return SynthesisResult(
                    audio=np.zeros(2400, dtype=np.float32),
                    sample_rate=24000,
                    duration_ms=100,
                )

        fake_setup = pw._SynthesisSetup(
            profile={"engine": "dummy", "pronunciation_dict": {"まはー": "マハー"}},
            cfg={},
            effective={"_enabled": True, "auto_speak": False},
            engine_name="dummy",
            engine=_DummyEngine(),  # type: ignore[arg-type]
            params={},
            ref_audio=None,
            ref_text=None,
        )

        with (
            patch.object(pw, "_get_effective_params", return_value={"_enabled": True, "auto_speak": False}),
            patch.object(pw._worker, "_prepare_synthesis", return_value=fake_setup),
            patch.object(pw._worker, "_save_wav") as mock_save,
            patch.object(pw, "_notify_audio_ready") as mock_notify,
        ):
            res_with_dict = pw.synthesize_wav(
                persona_id="air_city_a",
                text="**まはー**、おはよう https://example.com",
                apply_pronunciation=True,
                clean_text=True,
                timeout=10.0,
            )
            res_without_dict = pw.synthesize_wav(
                persona_id="air_city_a",
                text="まはー、おはよう",
                apply_pronunciation=False,
                clean_text=True,
                timeout=10.0,
            )

        self.assertTrue(res_with_dict.wav_bytes.startswith(b"RIFF"))
        self.assertEqual(res_with_dict.engine_name, "dummy")
        self.assertEqual(res_with_dict.sample_rate, 24000)
        self.assertAlmostEqual(res_with_dict.duration_sec, 0.1, places=3)
        self.assertEqual(res_with_dict.applied_text, "マハー、おはよう")
        self.assertEqual(res_without_dict.applied_text, "まはー、おはよう")
        self.assertEqual(captured_texts, ["マハー、おはよう", "まはー、おはよう"])
        # テスト合成は outputs/ 保存・通知を一切起こさない
        mock_save.assert_not_called()
        mock_notify.assert_not_called()

    def test_synthesize_wav_raises_when_addon_disabled(self):
        from tools.speak import playback_worker as pw

        with patch.object(pw, "_get_effective_params", return_value={"_enabled": False}):
            with self.assertRaises(pw.VoiceUnavailable):
                pw.synthesize_wav(persona_id="air_city_a", text="hello", timeout=5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
