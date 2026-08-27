"""FastAPI router for the saiverse-voice-tts addon.

The SAIVerse host auto-discovers ``expansion_data/<addon>/api_routes.py`` at
startup, imports the ``router`` attribute, and mounts it under
``/api/addon/<addon>``.

Endpoints exposed:

    GET  /api/addon/saiverse-voice-tts/audio/{message_id}
         → serves the fully synthesised WAV from disk.

    GET  /api/addon/saiverse-voice-tts/audio/{message_id}/stream
         → HTTP Chunked Transfer of the WAV as it is being synthesised.
           Browsers start playback as soon as the first chunk arrives,
           mirroring the server-side sounddevice streaming experience for
           remote (Tailscale) listeners.

Authentication: relies on the host's ``saiverse.addon_deps.get_manager``
dependency. Being able to resolve ``get_manager`` is, today, the effective
auth gate.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import queue
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

LOGGER = logging.getLogger(__name__)
_ADDON_NAME = "saiverse-voice-tts"

# ---------------------------------------------------------------------------
# Host-provided helpers (loaded lazily; absence is a fatal misconfiguration
# at request time but not at import time so the pack can still register the
# router when the addon framework is partially available).
# ---------------------------------------------------------------------------


def _get_metadata(message_id: str) -> Any:
    from saiverse.addon_metadata import get_metadata  # type: ignore
    return get_metadata(message_id=message_id, addon_name=_ADDON_NAME)


def _get_manager_dep() -> Any:
    """Return the host's ``get_manager`` dependency or a noop placeholder.

    The placeholder keeps routes importable on old host builds; real auth
    requires the host to provide ``saiverse.addon_deps.get_manager``.
    """
    try:
        from saiverse.addon_deps import get_manager  # type: ignore
        return get_manager
    except Exception:  # pragma: no cover
        async def _noop() -> None:
            return None
        return _noop


# Import the stream registry from the tools tree. SAIVerse の tool loader は
# サブディレクトリのツール (speak/schema.py) を ``tools._loaded.<subdir>``
# という名前空間で sys.modules に登録するので、同じ audio_stream モジュール
# インスタンスを参照するためにこの絶対パスで import する必要がある。
#
# ``tools.speak.audio_stream`` ではホストの tools パッケージ下に存在しないため
# 常に ImportError になり、stub fallback の ``get_queue`` が常に None を返して
# しまい、結果として /audio/{id}/stream エンドポイントが live-queue を見失って
# 404 相当の挙動になる (既知のバグ修正)。
def _audio_stream_module() -> Any:
    """Resolve the live audio_stream module instance at call time.

    Tool loader registers under ``tools._loaded.speak.audio_stream``; using
    sys.modules lookup avoids stale import references if the module is
    reloaded during dev.
    """
    import sys
    return sys.modules.get("tools._loaded.speak.audio_stream")


def _playback_worker_module() -> Any:
    """同様に playback_worker モジュールを sys.modules から取得。

    host の addon_external_loader が `tools.speak.*` への通常 import を
    block するので、tool loader 登録名 `tools._loaded.speak.playback_worker`
    を直接参照する。
    """
    import sys
    return sys.modules.get("tools._loaded.speak.playback_worker")


def subscribe_stream(message_id: str) -> Any:
    mod = _audio_stream_module()
    if mod is None:
        return None
    return mod.subscribe(message_id)


def has_active_stream(message_id: str) -> bool:
    mod = _audio_stream_module()
    if mod is None:
        return False
    return bool(mod.has_stream(message_id))


router = APIRouter()


# ---------------------------------------------------------------------------
# GET /audio/{message_id}
# ---------------------------------------------------------------------------
@router.get("/audio/{message_id}")
async def get_audio(
    message_id: str,
    manager: Any = Depends(_get_manager_dep()),
) -> FileResponse:
    """Serve the fully synthesised WAV for ``message_id``.

    Looks up ``audio_path`` from the host's ``addon_metadata`` store, keyed by
    ``(message_id, addon_name=saiverse-voice-tts, key=audio_path)``.
    """
    try:
        meta = _get_metadata(message_id)
    except Exception as exc:
        LOGGER.exception("addon_metadata lookup failed for %s", message_id)
        raise HTTPException(status_code=500, detail="metadata lookup failed") from exc

    if not meta or not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail="audio not yet available")

    # audio_file はバックエンドローカルの WAV ファイルパス。
    # audio_path はフロント向け URL なのでファイルオープンに使ってはならない。
    fs_path = meta.get("audio_file")
    if not fs_path:
        # 旧バージョン互換: audio_file 未設定だが audio_path が実パスになっている
        # ケース (生成後にキーを分離する前のデータ) はフォールバックとして許容する。
        legacy = meta.get("audio_path")
        if legacy and not str(legacy).startswith("/api/"):
            fs_path = legacy
    if not fs_path:
        raise HTTPException(status_code=404, detail="audio not yet available")

    path = Path(str(fs_path))
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio file missing on disk")

    # filename を渡すと FastAPI が Content-Disposition: attachment をデフォルトで
    # 付与し、ブラウザが <audio> で再生せずダウンロード扱いしてしまう。
    # content_disposition_type="inline" を明示してインライン再生可能にする。
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=path.name,
        content_disposition_type="inline",
    )


# ---------------------------------------------------------------------------
# GET /audio/{message_id}/stream
# ---------------------------------------------------------------------------
#: consumer queue を 1 回待つときの上限 (秒)。
#:
#: 無期限の ``q.get()`` を ``run_in_executor`` に渡してはならない。渡すと worker
#: thread は「作業中」のまま executor へ戻らず、次の二つが同時に壊れる:
#:
#:   1. クライアントが切断してリクエストが cancel されても、待っている thread は
#:      残り続ける (別 thread で走る同期処理に asyncio の cancel は届かない)。
#:   2. インタプリタ終了時、``concurrent.futures`` が ``atexit`` より前に走らせる
#:      ``_python_exit`` がその thread を join し続けるため、ホストが ``atexit``
#:      に登録した後始末に到達できず、プロセスが終了しなくなる。
#:
#: ``close_stream`` まで到達しなかったストリーム — 合成が始まらないまま
#: subscribe された placeholder (``audio_stream.subscribe`` の subscribe-before-open)
#: や、途中で失敗した合成 — が 1 本でも残れば終了印は永久に来ないので、
#: 上の 2 は「いつか起きる」ではなく「起きる」。実際 2026-08-27 に SAIVerse
#: バックエンドが Ctrl+C で終了せず、この待ちが 3 本残っていた。
#:
#: 待ちを刻めば worker thread は毎周 executor へ返るので、停止シグナルも cancel も
#: 届くようになる。同じ理由から ``playback_worker`` の job queue も期限付きで待つ。
_CONSUMER_POLL_INTERVAL = 0.5


async def _stream_body(message_id: str) -> AsyncIterator[bytes]:
    """Yield MP3 bytes from the in-process broadcast stream as they arrive.

    Each call to this generator subscribes a new private consumer queue to
    the broadcast. The queue is seeded with frames already emitted so that
    multiple clients (browser + retry + curl 等) can all replay the full
    stream from the start.

    The wait for each chunk is bounded by ``_CONSUMER_POLL_INTERVAL`` so that
    the executor thread is never pinned by a stream that never closes.
    """
    q = subscribe_stream(message_id)
    if q is None:
        # Race: stream was discarded between the pre-check in stream_audio
        # and this generator starting. End the stream so the client retries
        # via the non-streaming audio_path URL.
        LOGGER.warning("[stream_body] msg=%s stream no longer available", message_id)
        return

    loop = asyncio.get_event_loop()
    next_chunk = functools.partial(q.get, True, _CONSUMER_POLL_INTERVAL)
    while True:
        try:
            chunk = await loop.run_in_executor(None, next_chunk)
        except queue.Empty:
            # まだ次の frame が来ていないだけ。ここで一度 await が返ることが
            # 重要で、cancel の受け口と worker thread の解放を兼ねている。
            continue
        if chunk is None:
            break
        yield chunk


@router.get("/audio/{message_id}/stream")
async def stream_audio(
    message_id: str,
    manager: Any = Depends(_get_manager_dep()),
) -> StreamingResponse:
    """Chunked MP3 stream consumed live while synthesis is in progress.

    MP3 (audio/mpeg) is used instead of WAV because:
      - Progressive WAV requires a ``data`` chunk size upfront; the
        0xFFFFFFFF "unknown size" hack is rejected by iOS Safari and some
        Chrome versions with "The operation is not supported".
      - MP3 is a pure concatenation of frames with no container-level size
        requirement, so chunked transfer works natively in all browsers.

    If no stream is open for this message_id (合成完了後 or 合成未開始),
    returns 404 so the client falls back to the non-streaming audio_path
    URL which serves the completed WAV file.
    """
    if not has_active_stream(message_id):
        raise HTTPException(
            status_code=404,
            detail="no active stream (use /audio/{message_id} for completed file)",
        )
    return StreamingResponse(
        _stream_body(message_id),
        media_type="audio/mpeg",
    )


# ---------------------------------------------------------------------------
# POST /regenerate
# ---------------------------------------------------------------------------
@router.post("/regenerate")
async def regenerate_audio(
    body: dict,
    manager: Any = Depends(_get_manager_dep()),
) -> dict:
    """既存メッセージの音声を再生成する。

    バブルボタン (id: regenerate_audio) のクリックで呼ばれる。host 側の
    AddonBubbleButtons.tsx は POST body に
    ``{"message_id", "text", "persona_id"}`` を載せて送る。pack 側はこの
    context をそのまま使って TTS を再 enqueue する (現在の pronunciation_dict
    が自動で適用される)。

    text や persona_id が body に無いケース (古いフロント等) のフォールバック
    として、message_id だけ渡されたら ``manager.building_histories`` を走査
    して text + persona_id を引き当てる。

    Returns:
        ``{"status": "enqueued", "job_id": <id>, "source": "body"|"history"}``
    """
    message_id = body.get("message_id")
    if not message_id:
        raise HTTPException(status_code=400, detail="message_id is required")

    text = body.get("text")
    persona_id = body.get("persona_id")
    source = "body"

    # フォールバック: body に context が無ければ host 内 building_histories を走査
    if not text or not persona_id:
        if manager is None:
            raise HTTPException(
                status_code=400,
                detail="text and persona_id required when manager is unavailable",
            )
        histories = getattr(manager, "building_histories", None) or {}
        found = None
        for _bid, history in histories.items():
            for msg in history:
                if msg.get("message_id") == message_id or msg.get("id") == message_id:
                    found = msg
                    break
            if found is not None:
                break
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"message_id {message_id} not found in building histories",
            )
        text = text or found.get("content") or ""
        persona_id = persona_id or found.get("persona_id")
        source = "history"

    if not text:
        raise HTTPException(status_code=400, detail="message text is empty; nothing to synthesize")
    if not persona_id:
        raise HTTPException(status_code=400, detail="persona_id is required (assistant message expected)")

    # pack 内 tool モジュールは host loader 経由で sys.modules に登録される。
    # 通常の `from tools.speak...` import は addon_external_loader で block される。
    pw = _playback_worker_module()
    if pw is None:
        LOGGER.error(
            "regenerate_audio: tools._loaded.speak.playback_worker not in sys.modules. "
            "host tool loader が pack のツールを未ロード?"
        )
        raise HTTPException(
            status_code=500,
            detail="playback_worker module unavailable; host tool loader not initialised",
        )
    job_id = pw.enqueue_tts(text=str(text), persona_id=str(persona_id), message_id=str(message_id))
    LOGGER.info(
        "regenerate_audio: enqueued job=%s message_id=%s persona=%s source=%s len=%d",
        job_id, message_id, persona_id, source, len(str(text)),
    )
    return {"status": "enqueued", "job_id": job_id, "source": source}


# ---------------------------------------------------------------------------
# POST /client_action_failed
# ---------------------------------------------------------------------------
@router.post("/client_action_failed")
async def client_action_failed(body: dict) -> dict:
    """Client action executor failure webhook.

    Called by the host when a browser-side action executor rejects (autoplay
    拒否、ネットワーク失敗等)。payload は {"action_id", "event",
    "error_reason", "message_id"}。

    現状は WARN ログに残すのみ。将来的にバブル再生ボタンを目立たせる等の
    フィードバック機構を足す場合はここに反応ロジックを書く。
    """
    try:
        action_id = body.get("action_id")
        event_name = body.get("event")
        reason = body.get("error_reason")
        message_id = body.get("message_id")
        LOGGER.warning(
            "client_action_failed: action=%s event=%s msg=%s reason=%s",
            action_id, event_name, message_id, reason,
        )
    except Exception:
        LOGGER.exception("client_action_failed: failed to parse body")
    return {"ok": True}


# ---------------------------------------------------------------------------
# GET /audio-devices
# ---------------------------------------------------------------------------
@router.get("/audio-devices")
async def list_audio_devices() -> dict:
    """List host machine audio output devices for the addon UI dropdown.

    Returned format matches the AddonParamSchema `options_endpoint` contract:
    ``{"options": ["<default>", "0: Speakers (Realtek)", ...]}``.
    The first entry is always ``"<default>"`` which maps to ``None`` on the
    backend (= sounddevice's default device).
    """
    options: list[str] = ["<default>"]
    try:
        import sounddevice as sd  # type: ignore
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            # 出力チャンネルを持つデバイスのみ（入力専用マイクを除外）
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", f"device{i}")).strip()
            options.append(f"{i}: {name}")
    except Exception:
        LOGGER.exception("failed to enumerate audio devices")
    return {"options": options}
