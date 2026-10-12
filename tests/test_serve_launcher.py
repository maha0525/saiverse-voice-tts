"""The synthesis server launcher (tools/speak/serve_launcher.py, behind serve.bat / serve.sh).

Design: docs/intent/synthesis_server.md「サーバーの起動」(verification 5).
``saiverse.addon_paths`` is replaced in ``sys.modules`` or by a stub SAIVerse
folder; the real ``~/.saiverse`` is not touched.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

_TESTS = Path(__file__).resolve().parent
_PACK_ROOT = _TESTS.parent
for _path in (_PACK_ROOT, _TESTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_engine_env import _NO_SAIVERSE, _fake_addon_paths  # noqa: E402
from tools.speak import serve_launcher  # noqa: E402

_LAUNCHER = _PACK_ROOT / "tools" / "speak" / "serve_launcher.py"


class LauncherPlanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        # The launcher adds the SAIVerse folder and tools/speak to sys.path
        saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), saved_path))

    def _env_dir(self, name):
        env_dir = self.root / "addon_install" / "saiverse-voice-tts" / "envs" / name
        env_dir.mkdir(parents=True)
        return env_dir

    def test_fails_when_saiverse_cannot_be_imported(self):
        with patch.dict(sys.modules, _NO_SAIVERSE), \
                patch.object(serve_launcher.subprocess, "Popen") as popen, \
                patch("sys.stderr") as stderr:
            self.assertEqual(serve_launcher.main(["--saiverse-root", str(self.root)]), 2)
        popen.assert_not_called()  # never falls back to the current Python
        message = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("saiverse.addon_paths", message)
        self.assertIn("--python", message)

    def test_fails_when_the_engine_environment_is_missing(self):
        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            with self.assertRaises(serve_launcher.LaunchError) as ctx:
                serve_launcher.plan(["--saiverse-root", str(self.root)])
        self.assertIn("gpt_sovits", str(ctx.exception))
        self.assertIn("--python", str(ctx.exception))

    def test_version_mismatch_is_an_error(self):
        self._env_dir("gpt_sovits")
        with patch.dict(sys.modules, _fake_addon_paths(self.root, env_python_error=True)):
            with self.assertRaises(serve_launcher.LaunchError) as ctx:
                serve_launcher.plan(["--saiverse-root", str(self.root)])
        self.assertIn("入れ直して", str(ctx.exception))

    def test_unknown_engine_is_rejected(self):
        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            with self.assertRaisesRegex(serve_launcher.LaunchError, "unknown engine"):
                serve_launcher.plan(["--saiverse-root", str(self.root), "--engine", "openai_tts"])

    def test_engine_environment_command_and_env(self):
        env_dir = self._env_dir("irodori")
        caller_env = {
            "PYTHONHOME": "/somewhere", "PYTHONPATH": "/saiverse",
            "CUDA_VISIBLE_DEVICES": "1", "PATH": "/usr/bin",
        }
        with patch.dict(sys.modules, _fake_addon_paths(self.root)), \
                patch.dict(os.environ, caller_env):
            command, env = serve_launcher.plan([
                "--saiverse-root", str(self.root), "--engine", "irodori",
                "--host", "0.0.0.0", "--port", "9000", "--config", '{"device": "cpu"}',
            ])
        self.assertEqual(Path(command[0]), env_dir / "Scripts" / "python.exe")
        self.assertEqual(Path(command[1]), serve_launcher.SERVER_PATH)
        self.assertEqual(command[2:6], [
            "--engine", "irodori",
            "--data-dir", str((self.root / "addon_data" / "saiverse-voice-tts" / "server").resolve()),
        ])
        self.assertEqual(command[6:], ["--host", "0.0.0.0", "--port", "9000", "--config", '{"device": "cpu"}'])
        assert env is not None
        # The same environment the synthesis child process gets
        self.assertEqual(env["VIRTUAL_ENV"], str(env_dir))
        self.assertEqual(env["NLTK_DATA"], str(env_dir / "nltk_data"))
        self.assertEqual(env["HF_HOME"], str(env_dir / "hf_cache"))
        self.assertTrue(env["PATH"].startswith(str(env_dir / "Scripts") + os.pathsep))
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_explicit_data_dir_wins(self):
        self._env_dir("gpt_sovits")
        with patch.dict(sys.modules, _fake_addon_paths(self.root)):
            command, _env = serve_launcher.plan([
                "--saiverse-root", str(self.root), "--data-dir", str(self.root / "mine"),
            ])
        self.assertEqual(command[command.index("--data-dir") + 1], str((self.root / "mine").resolve()))

    def test_python_mode_requires_data_dir(self):
        with self.assertRaisesRegex(serve_launcher.LaunchError, "--data-dir is required"):
            serve_launcher.plan(["--python", sys.executable])

    def test_python_mode_skips_saiverse_and_keeps_the_caller_environment(self):
        with patch.dict(sys.modules, _NO_SAIVERSE):
            command, env = serve_launcher.plan([
                "--python", sys.executable, "--data-dir", str(self.root / "data"), "--port", "9001",
            ])
        self.assertIsNone(env)
        self.assertEqual(command, [
            sys.executable, str(serve_launcher.SERVER_PATH), "--engine", "gpt_sovits",
            "--data-dir", str((self.root / "data").resolve()), "--port", "9001",
        ])

    def test_missing_python_is_reported(self):
        with self.assertRaisesRegex(serve_launcher.LaunchError, "does not exist"):
            serve_launcher.plan(["--python", str(self.root / "nope.exe"), "--data-dir", str(self.root)])

    def test_exit_code_is_forwarded(self):
        proc = MagicMock()
        proc.wait.return_value = 7
        with patch.object(serve_launcher.subprocess, "Popen", return_value=proc) as popen, \
                patch("sys.stderr"):
            code = serve_launcher.main(["--python", sys.executable, "--data-dir", str(self.root)])
        self.assertEqual(code, 7)
        self.assertIsNone(popen.call_args.kwargs["env"])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _group_kwargs() -> dict:
    """Start the launcher in its own process group, so the server it starts can be stopped with it."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_tree(proc: subprocess.Popen) -> None:
    """Stop the launcher and the server it started."""
    import signal

    if proc.poll() is None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if os.name == "nt" and proc.poll() is None:
        # taskkill is slow (about 10 s), so it is only the fallback
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait(timeout=30)


class LauncherProcessTests(unittest.TestCase):
    """The launcher as a real process, with a stub SAIVerse folder (not an
    editable install: saiverse is found only through the folder the launcher
    adds to sys.path)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _run(self, args):
        port = _free_port()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # never touch a GPU here
        env.pop("PYTHONPATH", None)
        proc = subprocess.Popen(
            [sys.executable, str(_LAUNCHER), *args, "--port", str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **_group_kwargs(),
        )
        self.addCleanup(self._assert_server_stopped, port)
        self.addCleanup(_stop_tree, proc)
        return proc, port

    def _assert_server_stopped(self, port):
        """No server is left running after the launcher is stopped."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                requests.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            except requests.RequestException:
                return
            time.sleep(0.2)
        self.fail(f"server on port {port} outlived the launcher")

    def _wait_health(self, proc, port):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.fail(f"launcher exited {proc.returncode}: {proc.stdout.read().decode(errors='replace')}")
            try:
                return requests.get(f"http://127.0.0.1:{port}/health", timeout=1).json()
            except requests.RequestException:
                time.sleep(0.2)
        self.fail("server did not answer /health")

    def test_saiverse_mode_finds_the_engine_environment_and_starts(self):
        # Stub SAIVerse: the "engine environment" Python is this test's Python.
        env_dir = self.root / "envs" / "gpt_sovits"
        env_dir.mkdir(parents=True)
        package = self.root / "saiverse_root" / "saiverse"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "addon_paths.py").write_text(textwrap.dedent(f"""
            from pathlib import Path
            class AddonEnvError(RuntimeError):
                pass
            def get_addon_env_dir(addon_name, env_name):
                return Path({str(self.root / "envs")!r}) / env_name
            def get_addon_env_python(addon_name, env_name):
                return Path({sys.executable!r})
            def get_addon_data_dir(addon_name):
                path = Path({str(self.root / "addon_data")!r}) / addon_name
                path.mkdir(parents=True, exist_ok=True)
                return path
        """), encoding="utf-8")
        proc, port = self._run(["--saiverse-root", str(self.root / "saiverse_root")])
        health = self._wait_health(proc, port)
        self.assertEqual((health["engine"], health["protocol"]), ("gpt_sovits", 2))
        # Default profile store: the addon data folder, created by the server
        self.assertTrue((self.root / "addon_data" / "saiverse-voice-tts" / "server" / "profiles").is_dir())

    def test_python_mode_starts_the_server(self):
        proc, port = self._run(["--python", sys.executable, "--data-dir", str(self.root / "data")])
        health = self._wait_health(proc, port)
        self.assertEqual((health["engine"], health["protocol"]), ("gpt_sovits", 2))
        self.assertTrue((self.root / "data" / "profiles").is_dir())


if __name__ == "__main__":
    unittest.main()
