"""音声合成を別プロセスで動かすための代理エンジン (本体プロセス側)。

``TTSEngine`` を実装し、実際の合成は子プロセス (subprocess_worker) に委譲する。
本体プロセスがどれだけ GIL を握られても、子プロセスは独立した GIL/CPU/GPU
割り当てで動くため、合成速度が本体負荷に左右されない。

ローカル合成のエンジン (gpt_sovits / irodori) は、どちらもこの代理エンジン経由で
動く。子プロセスは、アドオンカタログの導入で作られた **エンジン専用の Python 環境**
(``~/.saiverse/addon_install/saiverse-voice-tts/envs/<env名>/``) の Python で起動する。
GPT-SoVITS と Irodori-TTS のパッケージは本体の venv とも互いとも両立しないため、
本体プロセスの中では import できない (SAIVerse 本体
``docs/intent/addon_catalog_management.md``「導入時の質問と、アドオン専用の Python 環境」)。

起動に使う Python の決め方 (``resolve_worker_python``):

1. ``saiverse.addon_paths`` が import できない (スタンドアロン・テスト) → ``sys.executable``
2. 専用環境のフォルダが無い (旧来の setup.bat による手動導入。本体 venv に全部入り)
   → ``sys.executable``
3. 専用環境がある → その Python。本体の Python と版がずれていたら
   (``AddonEnvError``)、黙って本体の Python に落とさず ``WorkerEnvError`` で止める

設計判断は docs/out_of_process.md を参照:
- 通信 = 標準入出力 (長さ付きフレーム)
- 適用 = gpt_sovits / irodori は常に別プロセス (create_engine が本クラスを返す)
- 異常時 = 一定時間応答が無ければ失敗扱い + 子プロセス自動再起動
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from . import subprocess_ipc as ipc
from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

_ADDON_ID = "saiverse-voice-tts"

_WORKER_PATH = Path(__file__).resolve().parent / "subprocess_worker.py"

# 初回リクエストはモデルロード (CUDA, 数百秒かかることがある) を含むので長め。
# Irodori-TTS も初回はモデルの読み込み (未取得なら HF からのダウンロード) を
# 含むので、同じ上限を使う。
_FIRST_LOAD_TIMEOUT = float(os.environ.get("VOICE_TTS_FIRST_LOAD_TIMEOUT", "600"))
# 2 フレーム目以降 / ウォーム後の 1 フレームの上限。
_FRAME_TIMEOUT = float(os.environ.get("VOICE_TTS_FRAME_TIMEOUT", "120"))

_EOF = object()  # reader スレッドが EOF (子プロセス終了) を伝えるための番兵

# 子プロセスで動かすエンジン。値は (専用環境の名前, ストリーミング対応か)。
# 専用環境の名前は addon.json の setup.steps の ``env`` と一致させること。
WORKER_ENGINES: Dict[str, Tuple[str, bool]] = {
    # GPT-SoVITS はネイティブのストリーミング推論
    "gpt_sovits": ("gpt_sovits", True),
    # Irodori-TTS は文単位チャンキングによる疑似ストリーミング (IrodoriEngine)
    "irodori": ("irodori", True),
}


class WorkerEnvError(RuntimeError):
    """専用の Python 環境が使えない (本体の Python と版がずれた等)。

    合成の失敗として playback_worker のログに出る。本体の Python で代わりに
    動かすと、専用環境に入れたはずのパッケージが見つからない別の失敗になり、
    原因 (入れ直しが要る) が見えなくなるので、ここで止めて理由を言う。
    """


def resolve_worker_python(env_name: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """子プロセスを起動する Python と、子プロセスに渡す環境変数を返す。

    返値の環境変数が None のときは、親の環境をそのまま引き継ぐ (旧来どおり)。
    """
    try:
        from saiverse.addon_paths import (  # type: ignore
            AddonEnvError,
            get_addon_env_dir,
            get_addon_env_python,
        )
    except ImportError:
        # SAIVerse 本体の外 (スタンドアロン実行・テスト) — 旧来どおり
        return sys.executable, None

    env_dir = get_addon_env_dir(_ADDON_ID, env_name)
    if not env_dir.exists():
        # 専用環境が無い = 旧来の手動導入 (setup.bat が本体 venv に全部入れた形)
        return sys.executable, None

    try:
        python = get_addon_env_python(_ADDON_ID, env_name)
    except AddonEnvError as exc:
        raise WorkerEnvError(
            f"音声エンジンの専用の Python 環境 ({env_name}) が、いまの SAIVerse の "
            "Python と合わなくなっています。アドオン管理から voice-tts を入れ直して"
            f"ください。(詳細: {exc})"
        ) from exc

    env = os.environ.copy()
    # 本体側で PYTHONHOME が立っていると、venv の Python が本体の標準ライブラリを
    # 探しに行って起動に失敗するので外す。PYTHONPATH も外す — 本体のルートが
    # 入っていると本体の tools/ が子プロセスに持ち込まれ、GPT-SoVITS 内部の
    # `from tools.X import` が本体側に化ける (subprocess_worker の docstring の
    # tools 衝突回避が崩れる)。
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    bin_dir = Path(python).parent
    env["VIRTUAL_ENV"] = str(env_dir)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    # setup の step (scripts/setup_*_data.py) が取得したデータの置き場所。
    # どちらも専用環境の下なので、アンインストールで環境と一緒に消える。
    env["NLTK_DATA"] = str(env_dir / "nltk_data")
    env["HF_HOME"] = str(env_dir / "hf_cache")
    return str(python), env


class SubprocessTTSEngine(TTSEngine):
    """ローカル合成エンジンを子プロセスで動かす代理エンジン。"""

    def __init__(self, engine_name: str, engine_config: Dict[str, Any]):
        if engine_name not in WORKER_ENGINES:
            raise ValueError(f"engine {engine_name!r} has no subprocess worker")
        super().__init__(engine_config)
        self.name = engine_name
        self._env_name, self.supports_streaming = WORKER_ENGINES[engine_name]
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._errlog = None
        self._warm = False  # 子プロセスがモデルロードを終えて 1 度でも音を出したか

    # -- subprocess lifecycle ------------------------------------------

    def _open_errlog(self):
        """子プロセスの stderr を流す per-session ログファイルを開く。

        エンジンが stdout/stderr に吐く推論ログ (GPT-SoVITS の n/1500 進捗、seed 等)
        をここに残す。エンジンごとに別ファイル (``voice_tts_worker_<engine>.log``)。
        """
        log_dir: Optional[Path] = None
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                log_dir = Path(handler.baseFilename).parent
                break
        if log_dir is None:
            import tempfile

            log_dir = Path(tempfile.gettempdir())
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return open(
                log_dir / f"voice_tts_worker_{self.name}.log",
                "a", encoding="utf-8", errors="replace",
            )
        except Exception:
            return subprocess.DEVNULL

    def _reader_loop(self, proc: subprocess.Popen, q: "queue.Queue") -> None:
        out = proc.stdout
        try:
            while True:
                payload = ipc.read_frame(out)
                if payload is None:
                    break
                q.put(payload)
        finally:
            q.put(_EOF)

    def _start_worker(self) -> None:
        self._kill_worker()
        # 専用環境が壊れているときはここで WorkerEnvError (子プロセスは作らない)
        python, env = resolve_worker_python(self._env_name)
        try:
            cfg_json = json.dumps(self.config)
        except Exception:
            LOGGER.warning("voice-tts: engine_config is not JSON-serializable; using {}")
            cfg_json = "{}"

        self._errlog = self._open_errlog()
        self._proc = subprocess.Popen(
            [python, str(_WORKER_PATH), "--engine", self.name, "--config", cfg_json],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errlog,
            bufsize=0,
            env=env,
        )
        self._warm = False
        self._queue = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_loop,
            args=(self._proc, self._queue),
            name=f"voice-tts-proxy-reader-{self.name}",
            daemon=True,
        )
        self._reader.start()
        LOGGER.info(
            "voice-tts: %s synthesis subprocess started (pid=%s, python=%s)",
            self.name, self._proc.pid, python,
        )

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._start_worker()

    def _kill_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        if self._errlog not in (None, subprocess.DEVNULL):
            try:
                self._errlog.close()
            except Exception:
                pass
        self._errlog = None

    # -- request helpers -----------------------------------------------

    def _send(self, method: str, text: str, ref_audio, ref_text, params) -> None:
        try:
            ipc.write_frame(
                self._proc.stdin,  # type: ignore[union-attr]
                ipc.encode_request(method, text, ref_audio, ref_text, params),
            )
        except Exception as exc:
            self._kill_worker()
            raise RuntimeError(f"voice-tts worker: failed to send request: {exc}") from exc

    def _next_payload(self, timeout: float) -> bytes:
        try:
            payload = self._queue.get(timeout=timeout)
        except queue.Empty:
            self._kill_worker()
            raise RuntimeError(
                f"voice-tts worker: no response within {timeout}s (restarting subprocess)"
            )
        if payload is _EOF:
            self._kill_worker()
            raise RuntimeError("voice-tts worker exited unexpectedly")
        return payload  # type: ignore[return-value]

    # -- TTSEngine interface -------------------------------------------

    def synthesize_stream(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SynthesisChunk]:
        with self._lock:
            self._ensure_worker()
            self._send("stream", text, ref_audio, ref_text, params)
            timeout = _FRAME_TIMEOUT if self._warm else _FIRST_LOAD_TIMEOUT
            while True:
                payload = self._next_payload(timeout)
                timeout = _FRAME_TIMEOUT
                kind, value = ipc.decode_response(payload)
                if kind == "chunk":
                    self._warm = True
                    audio, sr = value
                    yield SynthesisChunk(audio=audio.copy(), sample_rate=sr)
                elif kind == "result":  # 念のため (stream では通常来ない)
                    self._warm = True
                    audio, sr, _dur = value
                    yield SynthesisChunk(audio=audio.copy(), sample_rate=sr)
                elif kind == "end":
                    return
                elif kind == "error":
                    raise RuntimeError(f"voice-tts worker: {value}")

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        with self._lock:
            self._ensure_worker()
            self._send("synthesize", text, ref_audio, ref_text, params)
            timeout = _FRAME_TIMEOUT if self._warm else _FIRST_LOAD_TIMEOUT
            result: Optional[SynthesisResult] = None
            while True:
                payload = self._next_payload(timeout)
                timeout = _FRAME_TIMEOUT
                kind, value = ipc.decode_response(payload)
                if kind == "result":
                    self._warm = True
                    audio, sr, dur = value
                    result = SynthesisResult(audio=audio.copy(), sample_rate=sr, duration_ms=dur)
                elif kind == "chunk":  # 念のため
                    self._warm = True
                    audio, sr = value
                    result = SynthesisResult(
                        audio=audio.copy(),
                        sample_rate=sr,
                        duration_ms=int(len(audio) / sr * 1000) if sr else 0,
                    )
                elif kind == "end":
                    break
                elif kind == "error":
                    raise RuntimeError(f"voice-tts worker: {value}")
            if result is None:
                raise RuntimeError("voice-tts worker: no audio produced")
            return result

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            if proc is not None:
                try:
                    if proc.stdin:
                        proc.stdin.close()  # stdin を閉じて worker に終了を促す
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if self._errlog not in (None, subprocess.DEVNULL):
                try:
                    self._errlog.close()
                except Exception:
                    pass
            self._errlog = None
