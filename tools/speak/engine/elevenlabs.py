"""ElevenLabs TTS engine (cloud, no GPU required, voice cloning supported).

ElevenLabs の Text-to-Speech API を呼び出す。GPT-SoVITS / Irodori と同様
にゼロショットボイスクローンを使えるが、本エンジン v1 では voice_id を
ユーザーが ElevenLabs ダッシュボードで作成 → addon UI でペルソナごとに
voice_id を貼り付ける運用 (将来 ref_audio から自動クローンする経路は
v2 で検討)。

特徴:
- ストリーミング対応 (HTTP chunked + ``output_format=pcm_24000``)。
- API key は addon param ``elevenlabs_api_key`` 優先、未設定時は環境変数
  ``ELEVENLABS_API_KEY`` を使う。
- 依存: ``httpx`` のみ (本体 requirements に既存)。

API リファレンス:
- POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream
- https://elevenlabs.io/docs/api-reference/streaming
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import httpx
import numpy as np

from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
_PCM_SAMPLE_RATE = 24_000  # output_format=pcm_24000 と一致
# eleven_turbo_v2_5 は ElevenLabs が非推奨にした (公式の Models ページ、
# 2026-11-23 確認)。後継として案内されている eleven_flash_v2_5 は
# 「機能的に同等」とされ、料金も同じ。
_DEFAULT_MODEL_ID = "eleven_flash_v2_5"
# Eleven v4 は Stability と Similarity だけを使い、Style のスライダーが無い
# (公式の Eleven v4 ページ)。v4 系には voice_settings をこの二つだけ送る。
_V4_VOICE_SETTING_KEYS = ("stability", "similarity_boost")
_DEFAULT_OUTPUT_FORMAT = "pcm_24000"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 1
_RETRY_BASE_SLEEP = 1.5

# voice_settings の既定値 (ElevenLabs 公式推奨)
_DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
}


class ElevenLabsEngine(TTSEngine):
    name = "elevenlabs"
    supports_streaming = True

    def __init__(self, engine_config: Dict[str, Any]):
        super().__init__(engine_config)
        self._api_key_legacy: Optional[str] = (
            (engine_config or {}).get("api_key") or None
        )
        self._timeout = float((engine_config or {}).get("timeout", 120.0))

    # -------------------------------------------------------------- helpers

    def _resolve_api_key(self) -> Optional[str]:
        """API key の優先順位: addon UI param → config/default.json → env。"""
        try:
            from saiverse.addon_config import get_params  # type: ignore
            params = get_params("saiverse-voice-tts")
            ui_key = (params.get("elevenlabs_api_key") or "").strip()
            if ui_key:
                return ui_key
        except Exception as exc:
            LOGGER.debug("elevenlabs: addon_config unavailable: %s", exc)
        if self._api_key_legacy:
            return self._api_key_legacy
        env = os.getenv("ELEVENLABS_API_KEY")
        return env or None

    @staticmethod
    def _resolve_voice_id(params: Optional[Dict[str, Any]]) -> Optional[str]:
        params = params or {}
        # 優先順位: profile.params.voice_id → profile.params.elevenlabs_voice_id
        # (UI から `elevenlabs_voice_id` で渡ってくる想定だが、registry.json で
        # 直接 voice_id と書きたいケースも許容)
        for key in ("voice_id", "elevenlabs_voice_id"):
            v = params.get(key)
            if v:
                return str(v).strip()
        return None

    @staticmethod
    def _resolve_model_id(params: Optional[Dict[str, Any]]) -> str:
        params = params or {}
        # 優先順位: profile.params.model_id (registry.json に直接書く場合)
        # → profile.params.elevenlabs_model (addon UI のペルソナ別設定) → 既定
        for key in ("model_id", "elevenlabs_model"):
            v = params.get(key)
            if v and str(v).strip():
                return str(v).strip()
        return _DEFAULT_MODEL_ID

    @staticmethod
    def _build_voice_settings(params: Dict[str, Any]) -> Dict[str, Any]:
        settings = dict(_DEFAULT_VOICE_SETTINGS)
        for k in ("stability", "similarity_boost", "style"):
            if k in params and params[k] is not None:
                try:
                    f = float(params[k])
                    if 0.0 <= f <= 1.0:
                        settings[k] = f
                except (TypeError, ValueError):
                    pass
        if "use_speaker_boost" in params and params["use_speaker_boost"] is not None:
            settings["use_speaker_boost"] = bool(params["use_speaker_boost"])
        return settings

    def _build_request_body(
        self,
        text: str,
        params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        params = params or {}
        model_id = self._resolve_model_id(params)
        voice_settings = self._build_voice_settings(params)
        if model_id.startswith("eleven_v4"):
            voice_settings = {
                k: voice_settings[k] for k in _V4_VOICE_SETTING_KEYS
            }
        body: Dict[str, Any] = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        return body

    def _stream_url(self, voice_id: str, output_format: str) -> str:
        return (
            f"{_API_BASE}/{voice_id}/stream"
            f"?output_format={output_format}"
        )

    def _post_streaming_pcm(
        self,
        voice_id: str,
        body: Dict[str, Any],
        api_key: str,
        output_format: str,
    ) -> Iterator[bytes]:
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        url = self._stream_url(voice_id, output_format)
        attempt = 0
        while True:
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json=body,
                    headers=headers,
                    timeout=self._timeout,
                ) as resp:
                    if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                        LOGGER.warning(
                            "elevenlabs: HTTP %d (attempt %d), retrying",
                            resp.status_code, attempt + 1,
                        )
                        attempt += 1
                        time.sleep(_RETRY_BASE_SLEEP * attempt)
                        continue
                    if resp.status_code >= 400:
                        # httpx.stream の応答は本文をまだ読んでいないので、
                        # 読まずに raise すると下の except で resp.text が読めず、
                        # ElevenLabs が返した理由 (402 の中身など) がログに残らない。
                        resp.read()
                    resp.raise_for_status()
                    for chunk in resp.iter_bytes():
                        if chunk:
                            yield chunk
                return
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = ""
                try:
                    detail = exc.response.text[:500]
                except Exception as read_exc:  # pragma: no cover
                    detail = f"(response body unavailable: {read_exc})"
                LOGGER.error(
                    "elevenlabs: API error HTTP %d voice_id=%s: %s",
                    status, voice_id, detail,
                )
                raise
            except httpx.HTTPError as exc:
                if attempt < _MAX_RETRIES:
                    LOGGER.warning(
                        "elevenlabs: network error %s (attempt %d), retrying",
                        exc, attempt + 1,
                    )
                    attempt += 1
                    time.sleep(_RETRY_BASE_SLEEP * attempt)
                    continue
                LOGGER.error("elevenlabs: network error after retry: %s", exc)
                raise

    @staticmethod
    def _pcm_bytes_to_audio_np(pcm: bytes) -> np.ndarray:
        if not pcm:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    # -------------------------------------------------------------- public

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        chunks: List[np.ndarray] = []
        for chunk in self.synthesize_stream(text, ref_audio, ref_text, params):
            chunks.append(chunk.audio)
        if not chunks:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(chunks)
        duration_ms = int(len(audio) / _PCM_SAMPLE_RATE * 1000) if audio.size else 0
        return SynthesisResult(
            audio=audio,
            sample_rate=_PCM_SAMPLE_RATE,
            duration_ms=duration_ms,
        )

    def synthesize_stream(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SynthesisChunk]:
        if ref_audio:
            LOGGER.debug(
                "elevenlabs: ref_audio is ignored in v1 (manual voice_id only). "
                "Auto-cloning will be added in a future version."
            )
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                "elevenlabs: API key not configured. Set addon param "
                "'elevenlabs_api_key' or environment variable 'ELEVENLABS_API_KEY'."
            )
        voice_id = self._resolve_voice_id(params)
        if not voice_id:
            raise RuntimeError(
                "elevenlabs: voice_id not configured for this persona. "
                "Set 'elevenlabs_voice_id' on the persona's addon settings."
            )
        body = self._build_request_body(text, params)
        output_format = str(
            (params or {}).get("output_format") or _DEFAULT_OUTPUT_FORMAT
        )
        if output_format != _DEFAULT_OUTPUT_FORMAT:
            LOGGER.warning(
                "elevenlabs: output_format=%s is non-PCM, sample rate detection "
                "and direct numpy conversion are unsupported. Falling back to %s.",
                output_format, _DEFAULT_OUTPUT_FORMAT,
            )
            output_format = _DEFAULT_OUTPUT_FORMAT
        LOGGER.debug(
            "elevenlabs: synthesize voice_id=%s model=%s len=%d",
            voice_id, body["model_id"], len(text),
        )
        # 50 ms ぶん (4800 byte) でバッファリングしてから yield (本体側
        # ストリーミング再生のオーバーヘッド削減)。
        # 重要: int16 PCM なので numpy 変換は 2 byte 境界に揃ってる必要がある。
        # HTTP chunked transfer は任意のバイト境界で fragment し得るので、
        # flush 時に奇数 byte が末尾に来たら次回まで持ち越す。
        buf = bytearray()
        flush_threshold = 4800
        for raw in self._post_streaming_pcm(voice_id, body, api_key, output_format):
            buf.extend(raw)
            if len(buf) >= flush_threshold:
                emit_len = len(buf) - (len(buf) % 2)
                if emit_len > 0:
                    audio = self._pcm_bytes_to_audio_np(bytes(buf[:emit_len]))
                    del buf[:emit_len]
                    yield SynthesisChunk(audio=audio, sample_rate=_PCM_SAMPLE_RATE)
        if buf:
            emit_len = len(buf) - (len(buf) % 2)
            if emit_len > 0:
                audio = self._pcm_bytes_to_audio_np(bytes(buf[:emit_len]))
                yield SynthesisChunk(audio=audio, sample_rate=_PCM_SAMPLE_RATE)
