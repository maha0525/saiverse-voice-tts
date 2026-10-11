"""GPT-SoVITS over HTTP: profile-based remote synthesis server.

Server contract (shared with other SAIVerse clients of the same server):

- ``GET  {url}/health``                → ``{"ok": true, ...}``
- ``GET  {url}/profiles/{profile_id}`` → ``{"exists": bool, "ref_text": ...}`` or 404
- ``PUT  {url}/profiles/{profile_id}`` multipart ``ref_audio`` + ``ref_text`` → ``{"ok": true}``
- ``POST {url}/synthesize`` JSON ``{"profile_id", "text"}`` → complete ``audio/wav``
"""
from __future__ import annotations

import io
import logging
import math
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlsplit

import numpy as np
import requests
import soundfile as sf

from .base import SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)


class RemoteGPTSoVITSEngine(TTSEngine):
    name = "gpt_sovits"
    # The base adapter yields one completed WAV. This preserves the worker's
    # sub_seq/finalize path; the HTTP server itself does not stream inference.
    supports_streaming = True

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

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        params = params or {}
        profile_id = str(
            params.get("persona_id") or self.config.get("profile_id")
            or self.config.get("persona_id") or ""
        ).strip()
        started = time.monotonic()
        try:
            if not profile_id:
                raise RuntimeError("remote GPT-SoVITS requires persona_id or a configured profile_id")
            if ref_audio and ref_text:
                self._ensure_profile(profile_id, ref_audio, ref_text)
            try:
                response = self._request(
                    "post", "/synthesize", json={"profile_id": profile_id, "text": text},
                )
            except RuntimeError:
                # The server may have lost or replaced the profile (restart,
                # another client); check it again on the next sentence.
                self._confirmed.pop(profile_id, None)
                raise
            with sf.SoundFile(io.BytesIO(response.content)) as wav:
                if wav.format not in {"WAV", "WAVEX"} or wav.frames == 0:
                    raise RuntimeError("remote GPT-SoVITS returned an empty or non-WAV recording")
                sample_rate = wav.samplerate
                # All playback/save consumers expect mono float32 PCM. Average
                # channels rather than flattening stereo into twice the frames.
                audio = wav.read(dtype="float32", always_2d=True).mean(axis=1, dtype=np.float32)
            result = SynthesisResult(
                audio=audio, sample_rate=sample_rate,
                duration_ms=int(len(audio) / sample_rate * 1000),
            )
            LOGGER.debug(
                "remote GPT-SoVITS synthesized profile=%s chars=%d duration=%dms elapsed=%.2fs",
                profile_id, len(text), result.duration_ms, time.monotonic() - started,
            )
            return result
        except (OSError, ValueError, sf.SoundFileError) as exc:
            LOGGER.error("remote GPT-SoVITS profile/file/WAV failure (profile=%s): %s", profile_id, exc)
            raise RuntimeError(f"remote GPT-SoVITS profile/file/WAV failure: {exc}") from exc
        except RuntimeError as exc:
            LOGGER.error("remote GPT-SoVITS synthesis failed (profile=%s): %s", profile_id, exc)
            raise

    def close(self) -> None:
        self._http.close()
