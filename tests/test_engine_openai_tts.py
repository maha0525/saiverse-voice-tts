"""Tests for OpenAITTSEngine.

httpx.stream をモックして:
- リクエスト body の組み立て (model / voice / response_format=pcm / speed / instructions)
- API key 解決の優先順位 (addon UI > env > config)
- ストリーミング → SynthesisChunk への変換
- 4xx / 5xx 時の挙動 (4xx は即例外、5xx は 1 回リトライ)
- voice の不正値で alloy フォールバック

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
    if "tools.speak.engine.openai_tts" in sys.modules:
        del sys.modules["tools.speak.engine.openai_tts"]
    return importlib.import_module("tools.speak.engine.openai_tts")


def _install_fake_addon_config(get_params_func) -> None:
    pkg = sys.modules.get("saiverse") or types.ModuleType("saiverse")
    sys.modules["saiverse"] = pkg
    fake = types.ModuleType("saiverse.addon_config")
    fake.get_params = get_params_func  # type: ignore[attr-defined]
    sys.modules["saiverse.addon_config"] = fake


def _uninstall_fake_addon_config() -> None:
    sys.modules.pop("saiverse.addon_config", None)


class _FakeStreamResponse:
    """httpx.stream のコンテキストマネージャ風の偽レスポンス。"""

    def __init__(self, status_code: int, chunks: List[bytes], text: str = ""):
        self.status_code = status_code
        self._chunks = chunks
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            req = httpx.Request("POST", "https://api.openai.com/v1/audio/speech")
            resp = httpx.Response(self.status_code, request=req, text=self.text)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=resp,
            )

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._chunks


def _make_pcm_bytes(num_samples: int = 1200, sample_value: int = 100) -> bytes:
    """num_samples 個の int16 サンプル (= num_samples * 2 bytes) を作る。"""
    return (np.full(num_samples, sample_value, dtype=np.int16)).tobytes()


class OpenAITTSEngineRequestBodyTests(unittest.TestCase):
    """_build_request_body の検証。"""

    def setUp(self):
        self.mod = _import_engine()
        self.eng = self.mod.OpenAITTSEngine({})

    def test_default_body(self):
        body = self.eng._build_request_body("hello", None)
        self.assertEqual(body["model"], "tts-1")
        self.assertEqual(body["voice"], "alloy")
        self.assertEqual(body["input"], "hello")
        self.assertEqual(body["response_format"], "pcm")
        self.assertNotIn("speed", body)
        self.assertNotIn("instructions", body)

    def test_voice_resolved_from_openai_voice_key(self):
        body = self.eng._build_request_body("hi", {"openai_voice": "nova"})
        self.assertEqual(body["voice"], "nova")

    def test_voice_resolved_from_voice_key(self):
        body = self.eng._build_request_body("hi", {"voice": "echo"})
        self.assertEqual(body["voice"], "echo")

    def test_unknown_voice_falls_back_to_alloy(self):
        body = self.eng._build_request_body("hi", {"voice": "made_up"})
        self.assertEqual(body["voice"], "alloy")

    def test_speed_in_range(self):
        body = self.eng._build_request_body("hi", {"speed": 1.5})
        self.assertEqual(body["speed"], 1.5)

    def test_speed_out_of_range_dropped(self):
        body = self.eng._build_request_body("hi", {"speed": 10.0})
        self.assertNotIn("speed", body)

    def test_instructions_passed_through(self):
        body = self.eng._build_request_body("hi", {"instructions": "calmly"})
        self.assertEqual(body["instructions"], "calmly")


class OpenAITTSEngineApiKeyTests(unittest.TestCase):
    """API key 優先順位の検証: addon UI > config legacy > env。"""

    def setUp(self):
        self.mod = _import_engine()

    def tearDown(self):
        _uninstall_fake_addon_config()
        os.environ.pop("OPENAI_API_KEY", None)

    def test_addon_ui_wins(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"openai_api_key": "ui-key"}
        )
        os.environ["OPENAI_API_KEY"] = "env-key"
        eng = self.mod.OpenAITTSEngine({"api_key": "config-key"})
        self.assertEqual(eng._resolve_api_key(), "ui-key")

    def test_config_then_env_fallback(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"openai_api_key": ""}
        )
        os.environ["OPENAI_API_KEY"] = "env-key"
        eng = self.mod.OpenAITTSEngine({"api_key": "config-key"})
        self.assertEqual(eng._resolve_api_key(), "config-key")

    def test_env_only(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {}
        )
        os.environ["OPENAI_API_KEY"] = "env-key"
        eng = self.mod.OpenAITTSEngine({})
        self.assertEqual(eng._resolve_api_key(), "env-key")

    def test_no_key_returns_none(self):
        _install_fake_addon_config(
            lambda addon, persona_id=None: {}
        )
        eng = self.mod.OpenAITTSEngine({})
        self.assertIsNone(eng._resolve_api_key())


class OpenAITTSEngineSynthesizeStreamTests(unittest.TestCase):
    """synthesize_stream / synthesize の HTTP 経路検証。"""

    def setUp(self):
        self.mod = _import_engine()
        _install_fake_addon_config(
            lambda addon, persona_id=None: {"openai_api_key": "test-key"}
        )

    def tearDown(self):
        _uninstall_fake_addon_config()

    def test_synthesize_aggregates_streamed_chunks(self):
        # 2 chunk 合計 4800 byte (2400 samples)
        chunks = [_make_pcm_bytes(1200, 100), _make_pcm_bytes(1200, 200)]
        fake_resp = _FakeStreamResponse(200, chunks)
        eng = self.mod.OpenAITTSEngine({})
        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            result = eng.synthesize("hello")
        self.assertEqual(mock_stream.call_args.kwargs["json"]["voice"], "alloy")
        self.assertEqual(mock_stream.call_args.kwargs["json"]["response_format"], "pcm")
        self.assertEqual(result.sample_rate, 24_000)
        self.assertEqual(result.audio.shape[0], 2400)

    def test_synthesize_stream_yields_synthesis_chunks(self):
        chunks = [_make_pcm_bytes(2400, 100)]  # 1 chunk = 4800 byte (flush 閾値ぴったり)
        fake_resp = _FakeStreamResponse(200, chunks)
        eng = self.mod.OpenAITTSEngine({})
        with patch("httpx.stream", return_value=fake_resp):
            received = list(eng.synthesize_stream("hi"))
        self.assertGreaterEqual(len(received), 1)
        self.assertEqual(received[0].sample_rate, 24_000)
        self.assertTrue(isinstance(received[0].audio, np.ndarray))

    def test_no_api_key_raises(self):
        _install_fake_addon_config(lambda addon, persona_id=None: {})
        os.environ.pop("OPENAI_API_KEY", None)
        eng = self.mod.OpenAITTSEngine({})
        with self.assertRaises(RuntimeError) as ctx:
            list(eng.synthesize_stream("hi"))
        self.assertIn("API key not configured", str(ctx.exception))

    def test_4xx_does_not_retry(self):
        fake_resp = _FakeStreamResponse(401, [], text="invalid_api_key")
        eng = self.mod.OpenAITTSEngine({})
        with patch("httpx.stream", return_value=fake_resp) as mock_stream:
            with self.assertRaises(Exception):
                list(eng.synthesize_stream("hi"))
        self.assertEqual(mock_stream.call_count, 1)

    def test_5xx_retries_once(self):
        # 1 回目 503、2 回目 200 で OK
        first = _FakeStreamResponse(503, [], text="busy")
        second = _FakeStreamResponse(200, [_make_pcm_bytes(100)])
        eng = self.mod.OpenAITTSEngine({})
        with patch("httpx.stream", side_effect=[first, second]) as mock_stream:
            with patch("time.sleep"):  # リトライ間 sleep をスキップ
                list(eng.synthesize_stream("hi"))
        self.assertEqual(mock_stream.call_count, 2)

    def test_odd_byte_chunks_are_handled(self):
        """HTTP chunked transfer で奇数バイトが flush タイミングに重なっても
        ``buffer size must be a multiple of element size`` で死なないこと。

        2401 byte (奇数) を 3 回送信 = 累計 7203 byte (奇数)。flush 閾値 4800
        を超えた時点で、累計 4802 byte 中 4802 byte (偶数) を emit、残り 1 byte
        を保持。次回 chunk で 2402 byte 加算され累計 2403 byte → 4800 未満
        なので保持。最後 1 chunk 後に flush →  累計 7203 のうち 7202 byte を
        emit、末尾 1 byte は捨てられる。
        """
        chunks = [
            (np.full(1200, 50, dtype=np.int16)).tobytes() + b"\x00",  # 2401
            (np.full(1200, 60, dtype=np.int16)).tobytes() + b"\x00",  # 2401
            (np.full(1200, 70, dtype=np.int16)).tobytes() + b"\x00",  # 2401
        ]
        fake_resp = _FakeStreamResponse(200, chunks)
        eng = self.mod.OpenAITTSEngine({})
        with patch("httpx.stream", return_value=fake_resp):
            received = list(eng.synthesize_stream("hi"))
        # 何らかの SynthesisChunk が受け取れていればバグは再発していない
        self.assertGreaterEqual(len(received), 1)
        total_samples = sum(c.audio.shape[0] for c in received)
        # 7203 byte 中 7202 byte (= 3601 samples) を numpy 化してるはず
        self.assertEqual(total_samples, 3601)


if __name__ == "__main__":
    unittest.main(verbosity=2)
