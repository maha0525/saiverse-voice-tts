"""音声合成を別プロセスで動かすための代理エンジン (本体プロセス側)。

``TTSEngine`` を実装し、実際の合成は子プロセス (subprocess_worker) に委譲する。
本体プロセスがどれだけ GIL を握られても、子プロセスは独立した GIL/CPU/GPU
割り当てで動くため、合成速度が本体負荷に左右されない。

設計判断は docs/out_of_process.md を参照:
- 通信 = 標準入出力 (長さ付きフレーム)
- 適用 = gpt_sovits は常に別プロセス (create_engine が本クラスを返す)
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
from typing import Any, Dict, Iterator, Optional

from . import subprocess_ipc as ipc
from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

_WORKER_PATH = Path(__file__).resolve().parent / "subprocess_worker.py"

# 初回リクエストはモデルロード (CUDA, 数百秒かかることがある) を含むので長め。
_FIRST_LOAD_TIMEOUT = float(os.environ.get("VOICE_TTS_FIRST_LOAD_TIMEOUT", "600"))
# 2 フレーム目以降 / ウォーム後の 1 フレームの上限。
_FRAME_TIMEOUT = float(os.environ.get("VOICE_TTS_FRAME_TIMEOUT", "120"))

_EOF = object()  # reader スレッドが EOF (子プロセス終了) を伝えるための番兵


class SubprocessGPTSoVITSEngine(TTSEngine):
    """gpt_sovits を子プロセスで動かす代理エンジン。"""

    name = "gpt_sovits"
    supports_streaming = True

    def __init__(self, engine_config: Dict[str, Any]):
        super().__init__(engine_config)
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._errlog = None
        self._warm = False  # 子プロセスがモデルロードを終えて 1 度でも音を出したか

    # -- subprocess lifecycle ------------------------------------------

    def _open_errlog(self):
        """子プロセスの stderr を流す per-session ログファイルを開く。

        GPT-SoVITS が stdout/stderr に吐く推論ログ (n/1500 進捗、seed 等) を
        ここに残す。従来 stdout のみで消えていた情報がファイルに残るようになる。
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
            return open(log_dir / "voice_tts_worker.log", "a", encoding="utf-8", errors="replace")
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
        try:
            cfg_json = json.dumps(self.config)
        except Exception:
            LOGGER.warning("voice-tts: engine_config is not JSON-serializable; using {}")
            cfg_json = "{}"

        self._errlog = self._open_errlog()
        self._proc = subprocess.Popen(
            [sys.executable, str(_WORKER_PATH), "--config", cfg_json],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errlog,
            bufsize=0,
        )
        self._warm = False
        self._queue = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_loop,
            args=(self._proc, self._queue),
            name="voice-tts-proxy-reader",
            daemon=True,
        )
        self._reader.start()
        LOGGER.info("voice-tts: synthesis subprocess started (pid=%s)", self._proc.pid)

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
