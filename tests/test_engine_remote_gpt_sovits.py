"""Remote TTS contract and isolated worker-to-WAV tests; no real HTTP/DB."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import soundfile as sf

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak.engine import create_engine  # noqa: E402
from tools.speak.engine.remote_gpt_sovits import RemoteGPTSoVITSEngine  # noqa: E402
from tools.speak.engine.subprocess_proxy import SubprocessTTSEngine  # noqa: E402


def _response(status=200, body=None, content=b""):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode() if body is not None else content
    response.url = "http://tts.invalid"
    return response


def _wav(audio=None):
    if audio is None:
        audio = np.full(240, 0.25, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class RemoteEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = Path(self.tmp.name) / "ref.wav"
        self.ref.write_bytes(_wav())
        self.http = MagicMock()
        session_patch = patch("requests.Session", return_value=self.http)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        self.engine = create_engine("gpt_sovits", {
            "gpt_sovits_remote_url": " http://tts.invalid/api/ ",
            "gpt_sovits_remote_timeout_sec": 7,
        })
        self.addCleanup(self.engine.close)
        self.http.get.return_value = _response(body={"exists": True})
        self.http.put.return_value = _response(body={"ok": True})
        self.http.post.return_value = _response(content=_wav())

    def _synthesize(self, **kwargs):
        return self.engine.synthesize(
            "こんにちは", str(self.ref), "参照音声です", {"persona_id": "test persona/1"}, **kwargs,
        )

    def test_missing_profile_is_uploaded_before_synthesis(self):
        self.http.get.return_value = _response(404)
        uploaded = []

        def upload(*args, **kwargs):
            uploaded.append(kwargs["files"]["ref_audio"][1].read())
            return _response(body={"ok": True})

        self.http.put.side_effect = upload
        result = self._synthesize()
        self.assertIsInstance(self.engine, RemoteGPTSoVITSEngine)
        self.http.get.assert_called_once_with(
            "http://tts.invalid/api/profiles/test%20persona%2F1", timeout=7,
        )
        self.assertEqual(uploaded, [self.ref.read_bytes()])
        self.assertEqual(self.http.put.call_args.kwargs["data"], {"ref_text": "参照音声です"})
        self.assertEqual(self.http.put.call_args.kwargs["timeout"], 7)
        self.assertTrue(self.http.put.call_args.kwargs["files"]["ref_audio"][1].closed)
        self.http.post.assert_called_once_with(
            "http://tts.invalid/api/synthesize",
            json={"profile_id": "test persona/1", "text": "こんにちは"}, timeout=7,
        )
        self.assertEqual((result.sample_rate, result.duration_ms), (24000, 10))
        self.assertEqual(result.audio.dtype, np.float32)
        np.testing.assert_allclose(result.audio, 0.25)

    def test_exists_false_also_registers(self):
        self.http.get.return_value = _response(body={"exists": False})
        self._synthesize()
        self.http.put.assert_called_once()

    def test_existing_profile_does_not_read_or_upload_reference(self):
        self.ref.unlink()
        self._synthesize()
        self.http.put.assert_not_called()

    def test_matching_server_profile_is_not_reuploaded(self):
        self.http.get.return_value = _response(body={"exists": True, "ref_text": "参照音声です"})
        self._synthesize()
        self.http.put.assert_not_called()

    def test_changed_ref_text_on_server_is_updated(self):
        self.http.get.return_value = _response(body={"exists": True, "ref_text": "古い書き起こし"})
        self._synthesize()
        self.http.put.assert_called_once()
        self.assertEqual(self.http.put.call_args.kwargs["data"], {"ref_text": "参照音声です"})

    def test_confirmed_profile_skips_lookup_on_next_sentence(self):
        self.http.get.return_value = _response(body={"exists": True, "ref_text": "参照音声です"})
        self._synthesize()
        self._synthesize()
        self.http.get.assert_called_once()
        self.assertEqual(self.http.post.call_count, 2)

    def test_replaced_reference_file_is_reuploaded(self):
        self.http.get.return_value = _response(body={"exists": True, "ref_text": "参照音声です"})
        self._synthesize()
        self.http.put.assert_not_called()
        self.ref.write_bytes(_wav(np.full(480, 0.5, dtype=np.float32)))
        self._synthesize()
        self.assertEqual(self.http.get.call_count, 2)
        self.http.put.assert_called_once()

    def test_failed_synthesis_forgets_confirmation(self):
        self.http.get.return_value = _response(body={"exists": True, "ref_text": "参照音声です"})
        self._synthesize()
        self.http.post.return_value = _response(404)
        with self.assertRaises(RuntimeError):
            self._synthesize()
        self.http.post.return_value = _response(content=_wav())
        self._synthesize()
        self.assertEqual(self.http.get.call_count, 2)

    def test_existing_server_profile_needs_no_local_reference(self):
        self.engine.synthesize("hello", params={"persona_id": "synthetic"})
        self.http.get.assert_not_called()
        self.http.put.assert_not_called()

    def test_configured_profile_and_persona_override(self):
        self.engine.config["profile_id"] = "configured"
        self.engine.synthesize("hello")
        self.assertEqual(self.http.post.call_args.kwargs["json"]["profile_id"], "configured")
        self.engine.synthesize("hello", params={"persona_id": "synthetic"})
        self.assertEqual(self.http.post.call_args.kwargs["json"]["profile_id"], "synthetic")

    def test_missing_id_rejects_before_http(self):
        with self.assertRaisesRegex(RuntimeError, "requires persona_id"):
            self.engine.synthesize("hello")
        self.http.post.assert_not_called()

    def test_health_contract(self):
        self.http.get.return_value = _response(body={"ok": True})
        self.assertEqual(self.engine.health(), {"ok": True})
        self.http.get.assert_called_once_with("http://tts.invalid/api/health", timeout=7)
        self.http.get.return_value = _response(body={"ok": False})
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            self.engine.health()

    def test_stereo_is_downmixed_without_doubling_duration(self):
        self.http.post.return_value = _response(content=_wav(np.tile([0.5, -0.25], (240, 1))))
        result = self._synthesize()
        self.assertEqual(result.audio.shape, (240,))
        self.assertEqual(result.duration_ms, 10)
        np.testing.assert_allclose(result.audio, 0.125)

    def test_complete_wav_adapter_yields_one_chunk(self):
        chunks = list(self.engine.synthesize_stream("hello", params={"persona_id": "synthetic"}))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sample_rate, 24000)

    def test_transport_failures_propagate_with_log(self):
        for method in ("get", "put", "post"):
            for error in (requests.Timeout("timed out"), requests.ConnectionError("offline")):
                with self.subTest(method=method, error=error):
                    self.http.get.return_value = _response(404)
                    getattr(self.http, method).side_effect = error
                    with self.assertLogs(self.engine.__module__, level="ERROR"):
                        with self.assertRaises(RuntimeError) as caught:
                            self._synthesize()
                    self.assertIs(caught.exception.__cause__, error)
                    getattr(self.http, method).side_effect = None

    def test_http_errors_are_not_missing_profiles(self):
        for method in ("get", "put", "post"):
            with self.subTest(method=method):
                self.http.get.return_value = _response(404)
                getattr(self.http, method).return_value = _response(503)
                with self.assertRaisesRegex(RuntimeError, "503"):
                    self._synthesize()
                self.http.put.return_value = _response(body={"ok": True})
                self.http.post.return_value = _response(content=_wav())

    def test_rejected_upload_prevents_synthesis(self):
        self.http.get.return_value = _response(404)
        self.http.put.return_value = _response(body={"ok": False})
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self._synthesize()
        self.http.post.assert_not_called()

    def test_invalid_profile_json_prevents_upload(self):
        for response in (_response(content=b"invalid"), _response(body=[])):
            self.http.get.return_value = response
            with self.assertRaisesRegex(RuntimeError, "JSON"):
                self._synthesize()
        self.http.put.assert_not_called()
        self.http.post.assert_not_called()

    def test_invalid_profile_exists_does_not_overwrite_server_profile(self):
        for body in ({}, {"exists": "false"}, {"exists": None}):
            self.http.get.return_value = _response(body=body)
            with self.assertRaisesRegex(RuntimeError, "boolean exists"):
                self._synthesize()
        self.http.put.assert_not_called()
        self.http.post.assert_not_called()

    def test_invalid_or_empty_audio_is_rejected(self):
        for audio in (b"", b"not WAV", _wav(np.zeros(0, dtype=np.float32))):
            self.http.post.return_value = _response(content=audio)
            with self.assertRaises(RuntimeError):
                self._synthesize()

    def test_missing_reference_file_has_clear_error(self):
        self.http.get.return_value = _response(404)
        self.ref.unlink()
        with self.assertRaisesRegex(RuntimeError, "profile/file/WAV"):
            self._synthesize()
        self.http.post.assert_not_called()

    def test_url_and_timeout_validation(self):
        for url in ("", "file:///tmp/tts", "http://server#fragment"):
            with self.assertRaises(ValueError):
                RemoteGPTSoVITSEngine({"gpt_sovits_remote_url": url})
        for timeout in (0, 121, float("nan"), True):
            with self.assertRaises(ValueError):
                RemoteGPTSoVITSEngine({
                    "gpt_sovits_remote_url": "http://tts.invalid",
                    "gpt_sovits_remote_timeout_sec": timeout,
                })

    def test_local_default_and_irodori_are_preserved(self):
        with patch.dict(os.environ, {"VOICE_TTS_IN_PROCESS": "0"}):
            for config in ({}, {"gpt_sovits_remote_url": "  "}):
                engine = create_engine("gpt_sovits", config)
                self.assertIsInstance(engine, SubprocessTTSEngine)
                engine.close()
            engine = create_engine("irodori", {"gpt_sovits_remote_url": "http://tts.invalid"})
            self.assertIsInstance(engine, SubprocessTTSEngine)
            engine.close()
        with patch.dict(os.environ, {"VOICE_TTS_IN_PROCESS": "1"}):
            engine = create_engine("gpt_sovits", {"gpt_sovits_remote_url": "http://tts.invalid"})
            self.assertIsInstance(engine, RemoteGPTSoVITSEngine)
            engine.close()


class RemoteWorkerTests(unittest.TestCase):
    def test_ui_settings_to_http_and_accumulated_wav(self):
        from tools.speak import playback_worker as pw
        from tools.speak import profiles, pronunciation_dict

        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            worker = pw._TTSWorker()
            worker._config_loaded = True
            worker._config = {"engines": {"gpt_sovits": {"device": "cpu"}}}
            addon_config = types.ModuleType("saiverse.addon_config")
            settings = {
                "engine": "gpt_sovits", "streaming": True, "server_side_playback": False,
                "gpt_sovits_remote_url": "http://first.invalid",
                "gpt_sovits_remote_timeout_sec": 9,
            }
            addon_config.get_params = lambda *args, **kwargs: dict(settings)
            stack.enter_context(patch.dict(sys.modules, {"saiverse.addon_config": addon_config}))
            stack.enter_context(patch.object(pw, "_worker", worker))
            stack.enter_context(patch.object(pw, "_out_dir", return_value=Path(tmp)))
            stack.enter_context(patch.object(pronunciation_dict, "apply", side_effect=lambda text, **kw: text))
            ready = stack.enter_context(patch.object(pw, "_notify_audio_ready"))
            stack.enter_context(patch.object(pw, "_notify_stream_ready"))
            no_audio = stack.enter_context(patch.object(pw, "notify_no_audio"))
            stream = stack.enter_context(patch.object(pw, "audio_stream"))
            http = MagicMock()
            http.post.return_value = _response(content=_wav())
            stack.enter_context(patch("requests.Session", return_value=http))
            # get_profile follows the real UI loader: a registered remote
            # persona does not need a local reference file or registry entry.
            self.assertIsNotNone(profiles.get_profile("synthetic-one"))
            for seq in (1, 2):
                job = pw._Job(
                    text=f"sentence {seq}", persona_id="synthetic-one", job_id=f"job-{seq}",
                    message_id="synthetic-message", sub_seq=seq, is_final=False,
                )
                worker._process(job)
            self.assertIn("synthetic-message", worker._message_states)
            self.assertFalse(list(Path(tmp).glob("*.wav")))
            worker._process(pw._Job(
                text="", persona_id="synthetic-one", job_id="final",
                message_id="synthetic-message", is_final=True,
            ))
            self.assertNotIn("synthetic-message", worker._message_states)
            audio, sr = sf.read(next(Path(tmp).glob("*.wav")))
            self.assertEqual((len(audio), sr), (480, 24000))
            ready.assert_called_once()
            stream.open_stream.assert_called_once()
            self.assertEqual(stream.push_chunk.call_count, 2)
            no_audio.assert_not_called()
            self.assertEqual(http.post.call_args.kwargs["json"]["profile_id"], "synthetic-one")
            first = worker._get_engine("gpt_sovits", settings)
            self.assertEqual(first.base_url, "http://first.invalid")
            self.assertIs(worker._get_engine("gpt_sovits", settings), first)
            settings["gpt_sovits_remote_url"] = "http://second.invalid"
            worker._process(pw._Job(
                text="another persona", persona_id="synthetic-two", job_id="job-other",
                message_id="synthetic-other-message",
            ))
            second = worker._get_engine("gpt_sovits", settings)
            self.assertIsNot(second, first)
            self.assertEqual(second.base_url, "http://second.invalid")
            self.assertEqual(http.post.call_args.args[0], "http://second.invalid/synthesize")
            self.assertEqual(http.post.call_args.kwargs["json"]["profile_id"], "synthetic-two")
            # Personas on different servers alternate: both engines stay open.
            http.close.assert_not_called()
            settings["gpt_sovits_remote_timeout_sec"] = 11
            third = worker._get_engine("gpt_sovits", settings)
            self.assertIsNot(third, second)
            self.assertEqual(third.timeout_sec, 11)
            # Only a timeout change on the same server replaces the engine.
            http.close.assert_called_once()
            settings["gpt_sovits_remote_url_override"] = "http://override.invalid"
            fourth = worker._get_engine("gpt_sovits", settings)
            self.assertIsNot(fourth, third)
            self.assertEqual(fourth.base_url, "http://override.invalid")
            settings["gpt_sovits_remote_url_override"] = ""
            settings["gpt_sovits_remote_url"] = ""
            with patch.dict(os.environ, {"VOICE_TTS_IN_PROCESS": "0"}):
                local = worker._get_engine("gpt_sovits", settings)
            self.assertIsInstance(local, SubprocessTTSEngine)
            local.close()

    def test_local_and_remote_personas_alternate_without_closing_local(self):
        from tools.speak import playback_worker as pw

        worker = pw._TTSWorker()
        worker._config_loaded = True
        worker._config = {"engines": {"gpt_sovits": {}}}
        local_engine = MagicMock()
        remote_engine = MagicMock()

        def fake_create(name, cfg):
            return remote_engine if cfg.get("gpt_sovits_remote_url") else local_engine

        local_settings = {"gpt_sovits_remote_url": "", "gpt_sovits_remote_url_override": ""}
        remote_settings = {"gpt_sovits_remote_url": "", "gpt_sovits_remote_url_override": "http://r.invalid"}
        with patch.object(pw, "create_engine", side_effect=fake_create) as create:
            for _ in range(3):
                self.assertIs(worker._get_engine("gpt_sovits", local_settings), local_engine)
                self.assertIs(worker._get_engine("gpt_sovits", remote_settings), remote_engine)
        self.assertEqual(create.call_count, 2)
        local_engine.close.assert_not_called()
        remote_engine.close.assert_not_called()

    def test_empty_ui_url_keeps_config_file_url(self):
        from tools.speak import playback_worker as pw

        worker = pw._TTSWorker()
        worker._config_loaded = True
        worker._config = {"engines": {"gpt_sovits": {"gpt_sovits_remote_url": "http://file.invalid"}}}
        with patch("requests.Session"):
            engine = worker._get_engine("gpt_sovits", {
                "gpt_sovits_remote_url": "", "gpt_sovits_remote_timeout_sec": 15,
            })
            self.assertIsInstance(engine, RemoteGPTSoVITSEngine)
            self.assertEqual(engine.base_url, "http://file.invalid")
            engine.close()


class RemoteProfileTests(unittest.TestCase):
    """profiles.get_profile with the UI loader for remote GPT-SoVITS personas."""

    def _get_profile(self, settings, registry):
        from tools.speak import profiles

        addon_config = types.ModuleType("saiverse.addon_config")
        addon_config.get_params = lambda *args, **kwargs: dict(settings)
        with patch.dict(sys.modules, {"saiverse.addon_config": addon_config}), \
                patch.object(profiles, "_load_registry", return_value=registry):
            return profiles.get_profile("synthetic")

    def test_override_url_alone_makes_a_server_profile(self):
        profile = self._get_profile({
            "engine": "gpt_sovits", "gpt_sovits_remote_url": "",
            "gpt_sovits_remote_url_override": "http://override.invalid",
        }, {})
        self.assertIsNotNone(profile)
        self.assertEqual(profile["engine"], "gpt_sovits")
        self.assertIsNone(profile["ref_audio"])
        self.assertNotIn("_server_profile_only", profile)
        self.assertNotIn("gpt_sovits_remote_url_override", profile["params"])

    def test_no_url_and_no_reference_is_not_a_profile(self):
        profile = self._get_profile({
            "engine": "gpt_sovits", "gpt_sovits_remote_url": "",
            "gpt_sovits_remote_url_override": "",
        }, {})
        self.assertIsNone(profile)

    def test_registry_persona_entry_beats_server_profile(self):
        profile = self._get_profile(
            {"engine": "gpt_sovits", "gpt_sovits_remote_url": "http://r.invalid"},
            {"synthetic": {"engine": "gpt_sovits", "ref_audio": "samples/own.wav", "ref_text": "own"},
             "_default": {"engine": "gpt_sovits", "ref_audio": "samples/default.wav", "ref_text": "d"}},
        )
        self.assertTrue(profile["ref_audio"].endswith("own.wav"))

    def test_server_profile_beats_registry_default(self):
        profile = self._get_profile(
            {"engine": "gpt_sovits", "gpt_sovits_remote_url": "http://r.invalid"},
            {"_default": {"engine": "gpt_sovits", "ref_audio": "samples/default.wav", "ref_text": "d"}},
        )
        self.assertIsNone(profile["ref_audio"])

    def test_ui_reference_still_wins(self):
        profile = self._get_profile(
            {"engine": "gpt_sovits", "gpt_sovits_remote_url": "http://r.invalid",
             "ref_audio": "/abs/ui.wav", "ref_text": "ui"},
            {"synthetic": {"engine": "gpt_sovits", "ref_audio": "samples/own.wav", "ref_text": "own"}},
        )
        self.assertEqual(Path(profile["ref_audio"]).name, "ui.wav")
        self.assertEqual(profile["ref_text"], "ui")


if __name__ == "__main__":
    unittest.main()
