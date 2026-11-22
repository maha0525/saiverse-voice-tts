"""Tests for ElevenLabsEngine.

httpx.stream をモックして:
- リクエスト URL の構築 (voice_id + output_format)
- リクエスト body (text / model_id / voice_settings)
- voice_settings の範囲外値スキップ
- API key 解決の優先順位 (addon UI > config > env)
- voice_id 未設定なら例外
- ストリーミング → SynthesisChunk
- 4xx は即例外、5xx は 1 回リトライ

を検証する。実 API は叩かない。
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from typing import Iterator, List
from unittest.mock import patch

import numpy as np

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


def _import_engine():
    import importlib
    if "tools.speak.engine.elevenlabs" in sys.modules:
        del sys.modules["tools.speak.engine.elevenlabs"]
    return importlib.import_module("tools.speak.engine.elevenlabs")


def _install_fake_addon_config(get_params_func) -> None:
    pkg = sys.modules.get("saiverse") or types.ModuleType("saiverse")
    sys.modules["saiverse"] = pkg
    fake = types.ModuleType("saiverse.addon_config")
    fake.get_params = get_params_func  # type: ignore[attr-defined]
    sys.modules["saiverse.addon_config"] = fake


def _uninstall_fake_addon_config() -> None:
    sys.modules.pop("saiverse.addon_config", None)


class _FakeStreamResponse:
    """httpx.stream の応答の代わり。本物と同じく、``read()`` を呼ぶまで本文は
    読めない (例外に付いた response の ``.text`` は ResponseNotRead になる)。"""

    def __init__(self, status_code: int, chunks: List[bytes], text: str = ""):
        self.status_code = status_code
        self._chunks = chunks
        self._text = text
        self._read = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        self._read = True
        return self._text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            req = httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/x/stream")
            if self._read:
                resp = httpx.Response(self.status_code, request=req, text=self._text)
            else:
                resp = httpx.Response(
                    self.status_code, request=req,
                    stream=httpx.ByteStream(self._text.encode("utf-8")),
                )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=resp,
            )

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._chunks


def _make_pcm_bytes(num_samples: int = 1200, sample_value: int = 100) -> bytes:
    return (np.full(num_samples, sample_value, dtype=np.int16)).tobytes()


class ElevenLabsRequestBuildTests(unittest.TestCase):
    def setUp(self):
        self.mod = _import_engine()
        self.eng = self.mod.ElevenLabsEngine({})

    def test_voice_id_resolved_from_voice_id_key(self):
        v = self.eng._resolve_voice_id({"voice_id": "abc123"})
        self.assertEqual(v, "abc123")

    def test_voice_id_resolved_from_elevenlabs_voice_id_key(self):
        v = self.eng._resolve_voice_id({"elevenlabs_voice_id": "abc123"})
        self.assertEqual(v, "abc123")

    def test_voice_id_priority(self):
        # voice_id が優先される
        v = self.eng._resolve_voice_id(
            {"voice_id": "first", "elevenlabs_voice_id": "second"}
        )
        self.assertEqual(v, "first")

    def test_voice_id_none_when_missing(self):
        self.assertIsNone(self.eng._resolve_voice_id({}))
        self.assertIsNone(self.eng._resolve_voice_id(None))

    def test_default_body(self):
        body = self.eng._build_request_body("hello", None)
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["model_id"], "eleven_flash_v2_5")
        self.assertIn("voice_settings", body)
        self.assertEqual(body["voice_settings"]["stability"], 0.5)
        self.assertEqual(body["voice_settings"]["similarity_boost"], 0.75)
        self.assertEqual(body["voice_settings"]["use_speaker_boost"], True)

    def test_voice_settings_in_range(self):
        body = self.eng._build_request_body(
            "x", {"stability": 0.3, "similarity_boost": 0.9, "style": 0.2}
        )
        self.assertEqual(body["voice_settings"]["stability"], 0.3)
        self.assertEqual(body["voice_settings"]["similarity_boost"], 0.9)
        self.assertEqual(body["voice_settings"]["style"], 0.2)

    def test_voice_settings_out_of_range_dropped(self):
        body = self.eng._build_request_body(
            "x", {"stability": 5.0}  # invalid > 1.0
        )
        # 既定値が維持される
        self.assertEqual(body["voice_settings"]["stability"], 0.5)

    def test_use_speaker_boost_override(self):
        body = self.eng._build_request_body("x", {"use_speaker_boost": False})
        self.assertEqual(body["voice_settings"]["use_speaker_boost"], False)

    def test_model_id_override(self):
        body = self.eng._build_request_body(
            "x", {"model_id": "eleven_multilingual_v2"}
        )
        self.assertEqual(body["model_id"], "eleven_multilingual_v2")

    def test_model_from_ui_param(self):
        body = self.eng._build_request_body("x", {"elevenlabs_model": "eleven_v3"})
        self.assertEqual(body["model_id"], "eleven_v3")

    def test_model_id_wins_over_ui_param(self):
        body = self.eng._build_request_body(
            "x", {"model_id": "eleven_multilingual_v2", "elevenlabs_model": "eleven_v3"}
        )
        self.assertEqual(body["model_id"], "eleven_multilingual_v2")

    def test_blank_ui_param_falls_back_to_default(self):
        body = self.eng._build_request_body("x", {"elevenlabs_model": "  "})
        self.assertEqual(body["model_id"], "eleven_flash_v2_5")

    def test_v4_sends_only_stability_and_similarity(self):
        body = self.eng._build_request_body(
            "x", {"elevenlabs_model": "eleven_v4", "stability": 0.3, "style": 0.4}
        )
        self.assertEqual(body["model_id"], "eleven_v4")
        self.assertEqual(
            body["voice_settings"], {"stability": 0.3, "similarity_boost": 0.75}
        )

    def test_manifest_default_matches_engine_default(self):
        import json
        manifest = json.loads((Path(__file__).resolve().parent.parent / "addon.json").read_text(encoding="utf-8"))
        param = next(p for p in manifest["params_schema"] if p["key"] == "elevenlabs_model")
        self.assertEqual(param["default"], self.mod._DEFAULT_MODEL_ID)
        self.assertIn(self.mod._DEFAULT_MODEL_ID, param["options"])
        self.assertTrue(param["persona_configurable"])

    def test_stream_url_format(self):
        url = self.eng._stream_url("abc123", "pcm_24000")
        self.assertEqual(
            url,
            "https://api.elevenlabs.io/v1/text-to-speech/abc123/stream?output_format=pcm_24000",
        )


class ElevenLabsApiKeyTests(unittest.TestCase):
    def setUp(self):
        self.mod = _import_engine()

    def tearDown(self):
        _uninstall_fake_addon_config()
        os.environ.pop("ELEVENLABS_API_KEY", None)

    def test_addon_ui_wins(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"elevenlabs_api_key": "ui-key"}
        )
        os.environ["ELEVENLABS_API_KEY"] = "env-key"
        eng = self.mod.ElevenLabsEngine({"api_key": "config-key"})
        self.assertEqual(eng._resolve_api_key(), "ui-key")

    def test_config_then_env_fallback(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"elevenlabs_api_key": ""}
        )
        os.environ["ELEVENLABS_API_KEY"] = "env-key"
        eng = self.mod.ElevenLabsEngine({"api_key": "config-key"})
        self.assertEqual(eng._resolve_api_key(), "config-key")

    def test_env_only(self):
        _install_fake_addon_config(lambda addon, persona_id=None: {})
        os.environ["ELEVENLABS_API_KEY"] = "env-key"
        eng = self.mod.ElevenLabsEngine({})
        self.assertEqual(eng._resolve_api_key(), "env-key")

    def test_no_key_returns_none(self):
        _install_fake_addon_config(lambda addon, persona_id=None: {})
        eng = self.mod.ElevenLabsEngine({})
        self.assertIsNone(eng._resolve_api_key())


class ElevenLabsSynthesizeTests(unittest.TestCase):
    def setUp(self):
        self.mod = _import_engine()
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"elevenlabs_api_key": "test-key"}
        )

    def tearDown(self):
        _uninstall_fake_addon_config()

    def test_synthesize_aggregates_streamed_chunks(self):
        chunks = [_make_pcm_bytes(1200, 100), _make_pcm_bytes(1200, 200)]
        fake_resp = _FakeStreamResponse(200, chunks)
        eng = self.mod.ElevenLabsEngine({})
        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            result = eng.synthesize("hello", params={"voice_id": "abc"})
        # URL に voice_id が入っている
        called_url = mock_stream.call_args.args[1]
        self.assertIn("abc/stream", called_url)
        self.assertEqual(result.sample_rate, 24_000)
        self.assertEqual(result.audio.shape[0], 2400)

    def test_no_voice_id_raises(self):
        eng = self.mod.ElevenLabsEngine({})
        with self.assertRaises(RuntimeError) as ctx:
            list(eng.synthesize_stream("hi", params={}))
        self.assertIn("voice_id", str(ctx.exception))

    def test_no_api_key_raises(self):
        _install_fake_addon_config(lambda addon, persona_id=None: {})
        os.environ.pop("ELEVENLABS_API_KEY", None)
        eng = self.mod.ElevenLabsEngine({})
        with self.assertRaises(RuntimeError) as ctx:
            list(eng.synthesize_stream("hi", params={"voice_id": "x"}))
        self.assertIn("API key not configured", str(ctx.exception))

    def test_4xx_does_not_retry(self):
        fake_resp = _FakeStreamResponse(401, [], text="bad key")
        eng = self.mod.ElevenLabsEngine({})
        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            with self.assertRaises(Exception):
                list(eng.synthesize_stream("hi", params={"voice_id": "x"}))
        self.assertEqual(mock_stream.call_count, 1)

    def test_4xx_logs_the_reason_elevenlabs_returned(self):
        reason = '{"detail":{"code":"paid_plan_required","message":"Free users cannot use library voices via the API."}}'
        fake_resp = _FakeStreamResponse(402, [], text=reason)
        eng = self.mod.ElevenLabsEngine({})
        with patch("httpx.stream", return_value=fake_resp):
            with self.assertLogs(self.mod.LOGGER, level="ERROR") as logs:
                with self.assertRaises(Exception):
                    list(eng.synthesize_stream("hi", params={"voice_id": "x"}))
        self.assertTrue(any("paid_plan_required" in line for line in logs.output))

    def test_5xx_retries_once(self):
        first = _FakeStreamResponse(503, [], text="busy")
        second = _FakeStreamResponse(200, [_make_pcm_bytes(100)])
        eng = self.mod.ElevenLabsEngine({})
        with patch("httpx.stream", side_effect=[first, second]) as mock_stream:
            with patch("time.sleep"):
                list(eng.synthesize_stream("hi", params={"voice_id": "x"}))
        self.assertEqual(mock_stream.call_count, 2)

    def test_non_pcm_output_format_falls_back(self):
        chunks = [_make_pcm_bytes(100)]
        fake_resp = _FakeStreamResponse(200, chunks)
        eng = self.mod.ElevenLabsEngine({})
        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            list(eng.synthesize_stream(
                "hi",
                params={"voice_id": "x", "output_format": "mp3_44100_128"},
            ))
        called_url = mock_stream.call_args.args[1]
        # output_format は pcm_24000 にフォールバック
        self.assertIn("pcm_24000", called_url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
