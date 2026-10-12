"""GPT-SoVITS over HTTP: profile-based remote synthesis server.

Client of the Voice TTS speech synthesis server (tools/speak/server.py) and of
older servers with the same base contract. Design:
docs/intent/synthesis_server.md.

Server contract (shared with other SAIVerse clients of the same server):

- ``GET  {url}/health``                → ``{"ok": true, ..., "protocol": 2}``
- ``GET  {url}/profiles/{profile_id}`` → ``{"exists": bool, "ref_text": ...}`` or 404
- ``PUT  {url}/profiles/{profile_id}`` multipart ``ref_audio`` + ``ref_text`` → ``{"ok": true}``
- ``POST {url}/synthesize`` JSON ``{"profile_id", "text"}`` → complete ``audio/wav``

A server whose ``/health`` reports ``protocol`` >= 2 also takes ``"params"``
(the persona's synthesis params) and ``"stream": true``, and then answers with
the same length-prefixed frames as the local synthesis child process
(subprocess_ipc): audio chunks as they are synthesized, then the end frame.
Older servers (no ``protocol``) get the base request and return one WAV; the
persona's params do not apply there.
"""
from __future__ import annotations

import io
import logging
import math
import mimetypes
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote, urlsplit

import numpy as np
import requests
import soundfile as sf

from . import subprocess_ipc as ipc
from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

# /health の protocol がこの値以上なら、調整値とストリームの応答に対応している
STREAM_PROTOCOL = 2

_EOF = object()  # reader スレッドが応答本文の終わりを伝える番兵


class RemoteGPTSoVITSEngine(TTSEngine):
    name = "gpt_sovits"
    # Protocol-2 servers stream chunks as they are synthesized; older servers
    # return one completed WAV, which is yielded as a single chunk. Either way
    # the worker's sub_seq/finalize path concatenates per sentence.
    supports_streaming = True
    # The reference audio passed to synthesize() is stored on the server under
    # the persona's ID, so the worker must never pass the shared sample voice.
    stores_reference_by_persona = True

    def __init__(self, engine_config: Dict[str, Any]):
        super().__init__(engine_config)
        self.base_url = str(self.config.get("gpt_sovits_remote_url") or "").strip().rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("gpt_sovits_remote_url must be an HTTP(S) server URL")
        raw_timeout = self.config.get("gpt_sovits_remote_timeout_sec", 15)
        self.timeout_sec = float(raw_timeout)
        if isinstance(raw_timeout, bool) or not math.isfinite(self.timeout_sec) or not 1 <= self.timeout_sec <= 120:
            raise ValueError("gpt_sovits_remote_timeout_sec must be between 1 and 120")
        self._http = requests.Session()
        # profile_id -> signature of the local reference last confirmed to be
        # on the server. Lets every sentence skip the profile round-trip while
        # still re-uploading when the persona's reference audio/text changes.
        self._confirmed: Dict[str, Tuple[str, int, int, str]] = {}
        # The server's /health protocol, checked before the first synthesis and
        # forgotten whenever a synthesis fails (a different server may answer
        # at the same URL after a restart).
        self._protocol: Optional[int] = None
        self._warned_old_protocol = False
        LOGGER.debug("remote GPT-SoVITS engine initialized (timeout=%.1fs)", self.timeout_sec)

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = getattr(self._http, method)(
                f"{self.base_url}{path}", timeout=self.timeout_sec, **kwargs,
            )
            # A missing profile is the only non-success HTTP response that is
            # part of the protocol, rather than a failed request.
            if method == "get" and path.startswith("/profiles/") and response.status_code == 404:
                return response
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            LOGGER.error("remote GPT-SoVITS %s %s failed: %s", method.upper(), path, exc)
            raise RuntimeError(f"remote GPT-SoVITS {method.upper()} {path} failed: {exc}") from exc

    @staticmethod
    def _json(response: requests.Response) -> Dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("remote GPT-SoVITS returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise RuntimeError("remote GPT-SoVITS returned a non-object JSON response")
        return body

    def health(self) -> Dict[str, Any]:
        try:
            body = self._json(self._request("get", "/health"))
            if body.get("ok") is not True:
                raise RuntimeError("remote GPT-SoVITS server is not ready")
            return body
        except RuntimeError:
            LOGGER.error("remote GPT-SoVITS health check failed")
            raise

    def _ensure_profile(self, profile_id: str, ref_audio: str, ref_text: str) -> None:
        """Make the server's profile match the local reference audio/text.

        The local persona settings are the source of truth: the profile is
        (re)uploaded when the server lacks it, when the server's ref_text
        differs, or when the local reference changed since this engine last
        confirmed it. A missing local file is tolerated only while the server
        already holds a profile for this ID.
        """
        audio_path = Path(ref_audio)
        try:
            stat = audio_path.stat()
            signature: Optional[Tuple[str, int, int, str]] = (
                str(audio_path), stat.st_mtime_ns, stat.st_size, ref_text,
            )
        except OSError:
            signature = None
        if signature is not None and self._confirmed.get(profile_id) == signature:
            return

        path = f"/profiles/{quote(profile_id, safe='')}"
        response = self._request("get", path)
        server_text: Optional[str] = None
        exists = False
        if response.status_code != 404:
            body = self._json(response)
            exists = body.get("exists")
            if not isinstance(exists, bool):
                raise RuntimeError("remote GPT-SoVITS profile response requires boolean exists")
            if isinstance(body.get("ref_text"), str):
                server_text = body["ref_text"]

        previous = self._confirmed.get(profile_id)
        needs_upload = (
            not exists
            or (server_text is not None and server_text != ref_text)
            or (previous is not None and previous != signature)
        )
        if not needs_upload:
            if signature is not None:
                self._confirmed[profile_id] = signature
            return
        if signature is None and exists:
            LOGGER.warning(
                "remote GPT-SoVITS: reference file %s is unreadable; using the "
                "server's existing profile=%s as is", audio_path, profile_id,
            )
            return
        with audio_path.open("rb") as handle:
            response = self._request(
                "put", path,
                files={"ref_audio": (
                    audio_path.name, handle,
                    mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream",
                )},
                data={"ref_text": ref_text},
            )
        if self._json(response).get("ok") is not True:
            raise RuntimeError("remote GPT-SoVITS rejected the reference profile")
        if signature is not None:
            self._confirmed[profile_id] = signature
        LOGGER.info(
            "remote GPT-SoVITS %s profile=%s",
            "updated" if exists else "registered", profile_id,
        )

    # -- protocol / failure bookkeeping --------------------------------

    def _profile_id(self, params: Dict[str, Any]) -> str:
        return str(
            params.get("persona_id") or self.config.get("profile_id")
            or self.config.get("persona_id") or ""
        ).strip()

    def _server_protocol(self) -> int:
        """The server's protocol from /health, checked once until a synthesis fails.

        A /health that cannot be read fails the request (RuntimeError).
        """
        if self._protocol is None:
            raw = self.health().get("protocol")
            protocol = raw if isinstance(raw, int) and not isinstance(raw, bool) else 1
            if protocol < STREAM_PROTOCOL and not self._warned_old_protocol:
                self._warned_old_protocol = True
                LOGGER.warning(
                    "remote GPT-SoVITS server %s does not support synthesis params or "
                    "streaming (no protocol >= %d in /health); per-persona params such as "
                    "speed do not apply on this server",
                    self.base_url, STREAM_PROTOCOL,
                )
            self._protocol = protocol
        return self._protocol

    def _prepare(self, profile_id: str, ref_audio: Optional[str], ref_text: Optional[str]) -> int:
        if not profile_id:
            raise RuntimeError("remote GPT-SoVITS requires persona_id or a configured profile_id")
        protocol = self._server_protocol()
        if ref_audio and ref_text:
            self._ensure_profile(profile_id, ref_audio, ref_text)
        return protocol

    def _forget(self, profile_id: str) -> None:
        # The server may have lost or replaced the profile, or been replaced by
        # another server at the same URL (restart, another client); check both
        # again before the next sentence.
        self._confirmed.pop(profile_id, None)
        self._protocol = None

    def _file_failure(self, profile_id: str, exc: Exception) -> RuntimeError:
        """A local file / WAV / value failure, as the RuntimeError the worker expects."""
        self._forget(profile_id)
        LOGGER.error("remote GPT-SoVITS profile/file/WAV failure (profile=%s): %s", profile_id, exc)
        return RuntimeError(f"remote GPT-SoVITS profile/file/WAV failure: {exc}")

    def _synthesis_failure(self, profile_id: str, exc: RuntimeError) -> None:
        self._forget(profile_id)
        LOGGER.error("remote GPT-SoVITS synthesis failed (profile=%s): %s", profile_id, exc)

    # -- old servers: one complete WAV -----------------------------------

    def _synthesize_wav(self, profile_id: str, text: str) -> SynthesisResult:
        response = self._request(
            "post", "/synthesize", json={"profile_id": profile_id, "text": text},
        )
        with sf.SoundFile(io.BytesIO(response.content)) as wav:
            if wav.format not in {"WAV", "WAVEX"} or wav.frames == 0:
                raise RuntimeError("remote GPT-SoVITS returned an empty or non-WAV recording")
            sample_rate = wav.samplerate
            # All playback/save consumers expect mono float32 PCM. Average
            # channels rather than flattening stereo into twice the frames.
            audio = wav.read(dtype="float32", always_2d=True).mean(axis=1, dtype=np.float32)
        return SynthesisResult(
            audio=audio, sample_rate=sample_rate,
            duration_ms=int(len(audio) / sample_rate * 1000),
        )

    # -- protocol 2: params + streamed frames ----------------------------

    @staticmethod
    def _error_reason(response: requests.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("error"):
                return str(body["error"])
        except ValueError:
            pass
        return (response.text or response.reason or "")[:300]

    def _stream_chunks(
        self, profile_id: str, text: str, params: Dict[str, Any],
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """POST a streaming synthesis and yield (audio, sample_rate) per frame.

        The body is read on a separate thread into a queue, so the server can
        keep writing (and release its synthesis slot) while the consumer sleeps
        for playback flow control. Success requires the end frame; EOF before
        it, an error frame, or a read timeout (the configured timeout is the
        longest gap between data) fails the request.
        """
        body = {
            "profile_id": profile_id,
            "text": text,
            # persona_id is sent as profile_id; the rest are the persona's params
            "params": {k: v for k, v in params.items() if k != "persona_id"},
            "stream": True,
        }
        try:
            response = self._http.post(
                f"{self.base_url}/synthesize", json=body, timeout=self.timeout_sec, stream=True,
            )
        except requests.RequestException as exc:
            LOGGER.error("remote GPT-SoVITS POST /synthesize failed: %s", exc)
            raise RuntimeError(f"remote GPT-SoVITS POST /synthesize failed: {exc}") from exc
        try:
            if response.status_code >= 400:
                raise RuntimeError(
                    f"remote GPT-SoVITS POST /synthesize failed: HTTP {response.status_code}: "
                    f"{self._error_reason(response)}"
                )
            frames: "queue.Queue[Any]" = queue.Queue()
            threading.Thread(
                target=_read_frames, args=(response.raw, frames),
                name="voice-tts-remote-reader", daemon=True,
            ).start()
            # The reader's socket read times out after timeout_sec without data;
            # this wait only backs that up.
            backstop = self.timeout_sec + 5
            while True:
                try:
                    item = frames.get(timeout=backstop)
                except queue.Empty:
                    raise RuntimeError(
                        f"remote GPT-SoVITS sent no data for {backstop:.0f}s"
                    ) from None
                if item is _EOF:
                    raise RuntimeError("remote GPT-SoVITS closed the stream before its end")
                if isinstance(item, BaseException):
                    raise RuntimeError(f"remote GPT-SoVITS stream read failed: {item}") from item
                try:
                    kind, value = ipc.decode_response(item)
                except ValueError as exc:
                    raise RuntimeError(f"remote GPT-SoVITS sent an invalid frame: {exc}") from exc
                if kind in ("chunk", "result"):
                    # frombuffer arrays are read-only views of the frame bytes
                    yield value[0].copy(), int(value[1])
                elif kind == "error":
                    raise RuntimeError(f"remote GPT-SoVITS synthesis failed on the server: {value}")
                else:  # end
                    return
        finally:
            # Closing mid-stream drops the connection, so the server stops
            # synthesizing at its next write and serves the next request.
            response.close()

    # -- TTSEngine interface -------------------------------------------

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        params = params or {}
        profile_id = self._profile_id(params)
        started = time.monotonic()
        try:
            protocol = self._prepare(profile_id, ref_audio, ref_text)
            if protocol >= STREAM_PROTOCOL:
                # The same params as streaming playback, so the fallback
                # re-synthesis of a sentence sounds the same.
                result = _join(list(self._stream_chunks(profile_id, text, params)))
            else:
                result = self._synthesize_wav(profile_id, text)
        except (OSError, ValueError, sf.SoundFileError) as exc:
            raise self._file_failure(profile_id, exc) from exc
        except RuntimeError as exc:
            self._synthesis_failure(profile_id, exc)
            raise
        LOGGER.debug(
            "remote GPT-SoVITS synthesized profile=%s chars=%d duration=%dms elapsed=%.2fs",
            profile_id, len(text), result.duration_ms, time.monotonic() - started,
        )
        return result

    def synthesize_stream(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SynthesisChunk]:
        params = params or {}
        profile_id = self._profile_id(params)
        chunks: Optional[Iterator[Tuple[np.ndarray, int]]] = None
        try:
            protocol = self._prepare(profile_id, ref_audio, ref_text)
            if protocol < STREAM_PROTOCOL:
                result = self._synthesize_wav(profile_id, text)
                yield SynthesisChunk(audio=result.audio, sample_rate=result.sample_rate)
                return
            chunks = self._stream_chunks(profile_id, text, params)
            for audio, sample_rate in chunks:
                yield SynthesisChunk(audio=audio, sample_rate=sample_rate)
        except (OSError, ValueError, sf.SoundFileError) as exc:
            raise self._file_failure(profile_id, exc) from exc
        except RuntimeError as exc:
            self._synthesis_failure(profile_id, exc)
            raise
        finally:
            # Also when the consumer stops early: drop the connection now so the
            # server stops synthesizing for nobody.
            if chunks is not None:
                chunks.close()

    def close(self) -> None:
        self._http.close()


def _read_frames(raw: Any, frames: "queue.Queue[Any]") -> None:
    """Read length-prefixed frames from the HTTP body until it ends.

    Puts each payload, then ``_EOF`` at the end of the body, or the exception
    that stopped the read (connection reset, read timeout).
    """
    try:
        while True:
            payload = ipc.read_frame(raw)
            if payload is None:
                break
            frames.put(payload)
    except Exception as exc:
        frames.put(exc)
        return
    frames.put(_EOF)


def _join(chunks: List[Tuple[np.ndarray, int]]) -> SynthesisResult:
    voiced = [(chunk, sr) for chunk, sr in chunks if chunk.size]
    if not voiced:
        raise RuntimeError("remote GPT-SoVITS returned no audio")
    audio = [chunk for chunk, _sr in voiced]
    sample_rate = voiced[0][1]
    joined = np.concatenate(audio).astype(np.float32, copy=False)
    return SynthesisResult(
        audio=joined, sample_rate=sample_rate,
        duration_ms=int(len(joined) / sample_rate * 1000) if sample_rate else 0,
    )
