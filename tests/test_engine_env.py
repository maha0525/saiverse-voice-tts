"""アドオンカタログ対応 (v0.6.0) のテスト。

- 合成の子プロセスを起動する Python の決め方 (subprocess_proxy.resolve_worker_python)
  専用環境あり / なし / 本体の Python と版ずれ / SAIVerse 本体の外
- create_engine が gpt_sovits / irodori の両方を子プロセスの代理エンジンにすること
- device の決め方 (GPT-SoVITS の mps、Irodori-TTS の使えない device からの切り替え)
- 合成音声の保存先と、旧来の場所を指す記録の引き直し
- 設定ファイルのひな形のコピー (scripts/make_default_config.py)

``saiverse.addon_paths`` は ``sys.modules`` に差し替えたダミーで代用し、実機の
``~/.saiverse`` には触らない。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from tools.speak.engine import create_engine  # noqa: E402
from tools.speak.engine import subprocess_proxy  # noqa: E402
from tools.speak.engine.gpt_sovits import _resolve_device  # noqa: E402
from tools.speak.engine.irodori import _resolve_runtime_settings  # noqa: E402


class _FakeAddonEnvError(RuntimeError):
    pass


def _fake_addon_paths(root: Path, *, env_python_error: bool = False) -> dict:
    """``saiverse`` と ``saiverse.addon_paths`` のダミー (sys.modules 用)。"""
    mod = types.ModuleType("saiverse.addon_paths")
    mod.AddonEnvError = _FakeAddonEnvError  # type: ignore[attr-defined]

    def get_addon_env_dir(addon_name, env_name):
        return root / "addon_install" / addon_name / "envs" / env_name

    def get_addon_env_python(addon_name, env_name):
        if env_python_error:
            raise _FakeAddonEnvError("created with Python 3.12, now 3.13")
        return get_addon_env_dir(addon_name, env_name) / "Scripts" / "python.exe"

    def get_addon_data_dir(addon_name):
        path = root / "addon_data" / addon_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    mod.get_addon_env_dir = get_addon_env_dir  # type: ignore[attr-defined]
    mod.get_addon_env_python = get_addon_env_python  # type: ignore[attr-defined]
    mod.get_addon_data_dir = get_addon_data_dir  # type: ignore[attr-defined]
    return {"saiverse": types.ModuleType("saiverse"), "saiverse.addon_paths": mod}


# 本体の外 (import できない) を再現する: sys.modules の None は ImportError になる
_NO_SAIVERSE = {"saiverse": types.ModuleType("saiverse"), "saiverse.addon_paths": None}


class ResolveWorkerPythonTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_outside_saiverse_uses_current_python(self):
        with patch.dict(sys.modules, _NO_SAIVERSE):
            python, env = subprocess_proxy.resolve_worker_python("gpt_sovits")
        self.assertEqual(python, sys.executable)
        self.assertIsNone(env)

    def test_no_env_dir_means_legacy_manual_install(self):
        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            python, env = subprocess_proxy.resolve_worker_python("gpt_sovits")
        self.assertEqual(python, sys.executable)
        self.assertIsNone(env)

    def test_env_dir_uses_env_python_and_points_data_into_env(self):
        env_dir = self.root / "addon_install" / "saiverse-voice-tts" / "envs" / "irodori"
        env_dir.mkdir(parents=True)
        with patch.dict(sys.modules, _fake_addon_paths(self.root)), \
                patch.dict(os.environ, {"PYTHONHOME": "/somewhere", "PYTHONPATH": "/saiverse"}):
            python, env = subprocess_proxy.resolve_worker_python("irodori")
        self.assertEqual(Path(python), env_dir / "Scripts" / "python.exe")
        assert env is not None
        self.assertEqual(env["VIRTUAL_ENV"], str(env_dir))
        self.assertTrue(env["PATH"].startswith(str(env_dir / "Scripts") + os.pathsep))
        self.assertEqual(env["NLTK_DATA"], str(env_dir / "nltk_data"))
        self.assertEqual(env["HF_HOME"], str(env_dir / "hf_cache"))
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)

    def test_version_mismatch_is_an_error_not_a_silent_fallback(self):
        env_dir = self.root / "addon_install" / "saiverse-voice-tts" / "envs" / "gpt_sovits"
        env_dir.mkdir(parents=True)
        with patch.dict(sys.modules, _fake_addon_paths(self.root, env_python_error=True)):
            with self.assertRaises(subprocess_proxy.WorkerEnvError) as ctx:
                subprocess_proxy.resolve_worker_python("gpt_sovits")
        self.assertIn("入れ直して", str(ctx.exception))
        self.assertIn("gpt_sovits", str(ctx.exception))


class CreateEngineTests(unittest.TestCase):
    def test_local_engines_run_in_subprocess(self):
        for name in ("gpt_sovits", "irodori"):
            with self.subTest(engine=name), patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VOICE_TTS_IN_PROCESS", None)
                eng = create_engine(name, {})
                self.assertIsInstance(eng, subprocess_proxy.SubprocessTTSEngine)
                self.assertEqual(eng.name, name)
                self.assertTrue(eng.supports_streaming)
                self.assertEqual(eng._env_name, name)

    def test_unknown_worker_engine_rejected(self):
        with self.assertRaises(ValueError):
            subprocess_proxy.SubprocessTTSEngine("openai_tts", {})

    def test_worker_engines_match_addon_json_envs(self):
        """代理エンジンが使う専用環境の名前は、addon.json の step の env と一致する。"""
        import json

        manifest = json.loads((_PACK_ROOT / "addon.json").read_text(encoding="utf-8"))
        step_envs = {s["env"] for s in manifest["setup"]["steps"] if s.get("env")}
        proxy_envs = {env for env, _streaming in subprocess_proxy.WORKER_ENGINES.values()}
        self.assertEqual(step_envs, proxy_envs)


class GPTSoVITSDeviceTests(unittest.TestCase):
    def test_cuda(self):
        self.assertEqual(
            _resolve_device("cuda", cuda_available=True, mps_available=False), ("cuda", True)
        )

    def test_cuda_unavailable_falls_back_to_cpu_not_mps(self):
        self.assertEqual(
            _resolve_device("cuda", cuda_available=False, mps_available=True), ("cpu", False)
        )

    def test_mps_is_full_precision(self):
        self.assertEqual(
            _resolve_device("mps", cuda_available=False, mps_available=True), ("mps", False)
        )

    def test_mps_unavailable_falls_back_to_cpu(self):
        self.assertEqual(
            _resolve_device("mps", cuda_available=True, mps_available=False), ("cpu", False)
        )

    def test_cpu(self):
        self.assertEqual(
            _resolve_device("cpu", cuda_available=True, mps_available=True), ("cpu", False)
        )


class IrodoriRuntimeSettingsTests(unittest.TestCase):
    _TEMPLATE = {
        "device": "cuda",
        "model_precision": "bf16",
        "codec_device": "cuda",
        "codec_precision": "bf16",
    }

    def test_template_on_cuda_machine_is_unchanged(self):
        settings, notes = _resolve_runtime_settings(
            self._TEMPLATE, cuda_available=True, mps_available=False
        )
        self.assertEqual(settings, self._TEMPLATE)
        self.assertEqual(notes, [])

    def test_template_on_mac_moves_to_mps_fp32_and_cpu_codec(self):
        settings, notes = _resolve_runtime_settings(
            self._TEMPLATE, cuda_available=False, mps_available=True
        )
        self.assertEqual(
            settings,
            {
                "device": "mps",
                "model_precision": "fp32",
                "codec_device": "cpu",
                "codec_precision": "fp32",
            },
        )
        self.assertEqual(len(notes), 4)

    def test_template_without_gpu_moves_to_cpu(self):
        settings, _notes = _resolve_runtime_settings(
            self._TEMPLATE, cuda_available=False, mps_available=False
        )
        self.assertEqual(settings["device"], "cpu")
        self.assertEqual(settings["model_precision"], "fp32")

    def test_defaults_without_config(self):
        settings, notes = _resolve_runtime_settings({}, cuda_available=True, mps_available=False)
        self.assertEqual(
            settings,
            {
                "device": "cuda",
                "model_precision": "bf16",
                "codec_device": "cpu",
                "codec_precision": "fp32",
            },
        )
        self.assertEqual(notes, [])

    def test_explicit_mps_kept_when_available(self):
        settings, notes = _resolve_runtime_settings(
            {"device": "mps"}, cuda_available=False, mps_available=True
        )
        self.assertEqual(settings["device"], "mps")
        self.assertEqual(settings["model_precision"], "fp32")
        self.assertEqual(notes, [])


class OutputDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_inside_saiverse_uses_addon_data_outputs(self):
        from tools.speak import playback_worker as pw

        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            out = pw._out_dir()
        self.assertEqual(out, self.root / "addon_data" / "saiverse-voice-tts" / "outputs")

    def test_outside_saiverse_uses_legacy_dir(self):
        from tools.speak import playback_worker as pw

        with patch.dict(sys.modules, _NO_SAIVERSE):
            self.assertEqual(pw._out_dir(), pw.LEGACY_OUT_DIR)

    def test_recorded_legacy_path_is_found_in_outputs(self):
        import api_routes

        outputs = self.root / "addon_data" / "saiverse-voice-tts" / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "job123.wav").write_bytes(b"RIFF")
        legacy = self.root / "user_data" / "voice" / "out" / "job123.wav"  # 移行で無くなった
        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            self.assertEqual(api_routes._resolve_audio_file(str(legacy)), outputs / "job123.wav")
            self.assertIsNone(api_routes._resolve_audio_file(str(legacy.with_name("nope.wav"))))
            # wav 以外の名前では引き直さない
            self.assertIsNone(api_routes._resolve_audio_file(str(legacy.with_suffix(".txt"))))

    def test_recorded_path_that_exists_is_used_as_is(self):
        import api_routes

        wav = self.root / "somewhere" / "a.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"RIFF")
        with patch.dict(sys.modules, _NO_SAIVERSE):
            self.assertEqual(api_routes._resolve_audio_file(str(wav)), wav)


class MakeDefaultConfigTests(unittest.TestCase):
    def _load_script(self):
        spec = importlib.util.spec_from_file_location(
            "voice_tts_make_default_config", _PACK_ROOT / "scripts" / "make_default_config.py"
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def test_copies_templates_once_and_keeps_user_edits(self):
        mod = self._load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            (root / "voice_profiles").mkdir()
            (root / "config" / "default.json.template").write_text('{"a": 1}', encoding="utf-8")
            (root / "voice_profiles" / "registry.json.template").write_text(
                '{"b": 2}', encoding="utf-8"
            )
            mod._PACK_ROOT = root
            self.assertEqual(mod.main(), 0)
            self.assertEqual(
                (root / "config" / "default.json").read_text(encoding="utf-8"), '{"a": 1}'
            )
            self.assertTrue((root / "voice_profiles" / "registry.json").exists())
            self.assertTrue((root / "voice_profiles" / "samples" / "_default").is_dir())

            # 利用者の編集は、setup をやり直しても上書きしない
            (root / "config" / "default.json").write_text('{"edited": true}', encoding="utf-8")
            self.assertEqual(mod.main(), 0)
            self.assertEqual(
                (root / "config" / "default.json").read_text(encoding="utf-8"),
                '{"edited": true}',
            )


if __name__ == "__main__":
    unittest.main()
