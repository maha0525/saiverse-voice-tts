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
import datetime
import functools
import io
import json
import logging
import queue
import re
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

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


def _resolve_audio_file(fs_path: str) -> Path | None:
    """metadata の ``audio_file`` (合成時の絶対パス) から、いまの実ファイルを探す。

    v0.6.0 で合成音声の保存先を旧来の ``user_data/voice/out/`` から
    ``addon_data/saiverse-voice-tts/outputs/`` へ移した。旧来の場所の wav は
    SAIVerse 本体の起動時の移行 (``saiverse/addon_migrations.py``) が outputs/ へ
    移すが、移行前に記録した metadata の絶対パスは旧来の場所を指したまま残る。
    記録どおりの場所に無ければ、同じファイル名を outputs/ で探す (ファイル名は
    合成ジョブの ID なので、場所が変わっても一意)。
    """
    path = Path(fs_path)
    if path.is_file():
        return path
    if path.suffix.lower() != ".wav":
        return None
    try:
        from saiverse.addon_paths import get_addon_data_dir  # type: ignore
    except ImportError:
        return None
    candidate = get_addon_data_dir(_ADDON_NAME) / "outputs" / path.name
    return candidate if candidate.is_file() else None


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

    path = _resolve_audio_file(str(fs_path))
    if path is None:
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


# ---------------------------------------------------------------------------
# Voice test & batch WAV/ZIP synthesis helpers
# ---------------------------------------------------------------------------
_BATCH_MAX_ITEMS = 200
_SYNTH_TIMEOUT_SEC = 180.0
_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED_STEMS = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


def _sanitize_wav_filename(raw: Any, fallback_stem: str) -> str:
    """利用者指定のファイル名を安全な単一階層の `.wav` ファイル名へ整える。

    パストラバーサル (`/` `\\` `..`)、制御文字、Windows 禁止文字・予約デバイス名を
    無害化し、空になれば ``fallback_stem`` を使う。
    """
    text = str(raw or "").strip()
    if text:
        # パス区切りは取り除かず、全体を安全な文字へ置換して階層抜けを防ぐ
        text = text.replace("/", "_").replace("\\", "_")
        text = _INVALID_FILENAME_CHARS_RE.sub("_", text)
        text = text.replace("..", "_")
        text = re.sub(r"_+", "_", text)
        text = text.strip(" ._")
    if text.lower().endswith(".wav"):
        text = text[:-4].strip(" .")
    if not text:
        text = fallback_stem
    if len(text) > 100:
        text = text[:100].rstrip(" .") or fallback_stem
    if text.upper() in _WINDOWS_RESERVED_STEMS:
        text = f"{text}_"
    return f"{text}.wav"


def _unique_wav_filename(candidate: str, used_lower: set[str]) -> str:
    """同一 ZIP 内で重複しないファイル名を割り当てる (大文字小文字非依存)。"""
    lower = candidate.lower()
    if lower not in used_lower:
        used_lower.add(lower)
        return candidate
    stem = candidate[:-4] if lower.endswith(".wav") else candidate
    suffix = 2
    while True:
        alt = f"{stem}_{suffix}.wav"
        if alt.lower() not in used_lower:
            used_lower.add(alt.lower())
            return alt
        suffix += 1


def _content_disposition(kind: str, filename: str) -> str:
    """ASCII フォールバック + RFC 5987 UTF-8 filename* 付き Content-Disposition。"""
    ascii_fallback = re.sub(r"[^A-Za-z0-9._-]", "_", filename).strip("._")
    if not ascii_fallback:
        ascii_fallback = "voice.wav" if filename.lower().endswith(".wav") else "voices.zip"
    elif filename.lower().endswith(".wav") and not ascii_fallback.lower().endswith(".wav"):
        ascii_fallback += ".wav"
    elif filename.lower().endswith(".zip") and not ascii_fallback.lower().endswith(".zip"):
        ascii_fallback += ".zip"
    encoded = urllib.parse.quote(filename, safe="")
    return f'{kind}; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


def _map_synthesis_exception(exc: Exception, pw: Any) -> HTTPException:
    """``playback_worker.synthesize_wav`` の例外を適切な HTTPException に変換する。"""
    nothing_cls = getattr(pw, "NothingToVoice", None)
    unavail_cls = getattr(pw, "VoiceUnavailable", None)
    if nothing_cls is not None and isinstance(exc, nothing_cls):
        return HTTPException(status_code=400, detail=f"読み上げる文字がありません: {exc}")
    if unavail_cls is not None and isinstance(exc, unavail_cls):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, TimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=502, detail=f"音声合成に失敗しました: {exc}")


# ---------------------------------------------------------------------------
# POST /synthesize
# ---------------------------------------------------------------------------
@router.post("/synthesize")
async def synthesize_test_audio(
    body: dict,
    manager: Any = Depends(_get_manager_dep()),
) -> Response:
    """指定ペルソナの声でテキストを単発合成し、16-bit PCM WAV を返す。

    アドオン管理画面 (PersonaPanel) のテスト再生・WAV 保存・一括出力の
    各アイテム合成から呼ばれる。チャット履歴・吹き出し・SAIMemory・
    サーバー側スピーカー再生には一切触れない。
    """
    persona_id = str(body.get("persona_id") or "").strip()
    if not persona_id:
        raise HTTPException(status_code=400, detail="persona_id is required")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    apply_pronunciation = bool(body.get("apply_pronunciation", True))
    clean_text = bool(body.get("clean_text", True))
    raw_filename = body.get("filename")

    pw = _playback_worker_module()
    if pw is None or not hasattr(pw, "synthesize_wav"):
        LOGGER.error(
            "synthesize_test_audio: tools._loaded.speak.playback_worker.synthesize_wav unavailable"
        )
        raise HTTPException(
            status_code=500,
            detail="playback_worker module unavailable; host tool loader not initialised",
        )

    loop = asyncio.get_event_loop()
    call = functools.partial(
        pw.synthesize_wav,
        persona_id=persona_id,
        text=text,
        apply_pronunciation=apply_pronunciation,
        clean_text=clean_text,
        timeout=_SYNTH_TIMEOUT_SEC,
    )
    try:
        result = await loop.run_in_executor(None, call)
    except Exception as exc:
        raise _map_synthesis_exception(exc, pw) from exc

    fallback_stem = f"{persona_id}_test"
    download_name = _sanitize_wav_filename(raw_filename, fallback_stem)
    applied_preview = result.applied_text
    if len(applied_preview) > 500:
        applied_preview = applied_preview[:500] + "…"

    headers = {
        "Content-Disposition": _content_disposition("inline", download_name),
        "X-TTS-Engine": str(result.engine_name),
        "X-TTS-Duration-Sec": f"{result.duration_sec:.3f}",
        "X-TTS-Sample-Rate": str(result.sample_rate),
        "X-TTS-Applied-Text": urllib.parse.quote(applied_preview, safe=""),
        "Cache-Control": "no-store",
    }
    return Response(content=result.wav_bytes, media_type="audio/wav", headers=headers)


# ---------------------------------------------------------------------------
# POST /synthesize-batch
# ---------------------------------------------------------------------------
def _normalize_batch_items(
    raw_items: Any,
    default_persona_id: str,
) -> list[dict[str, str | None]]:
    """一括合成の ``items`` 配列を検証・正規化する。

    受け付ける要素:
      - ``"おはよう！"`` (文字列)
      - ``{"text": "おはよう！", "filename": "greeting", "persona_id": "..."}`` (オブジェクト)
    """
    if not isinstance(raw_items, list) or len(raw_items) == 0:
        raise HTTPException(status_code=400, detail="items must be a non-empty array")
    if len(raw_items) > _BATCH_MAX_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"items exceeds maximum of {_BATCH_MAX_ITEMS} (got {len(raw_items)})",
        )
    normalized: list[dict[str, str | None]] = []
    for idx, entry in enumerate(raw_items, start=1):
        if isinstance(entry, str):
            text = entry
            filename = None
            persona_id = default_persona_id
        elif isinstance(entry, dict):
            raw_text = entry.get("text")
            if not isinstance(raw_text, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"items[{idx - 1}].text must be a string",
                )
            text = raw_text
            raw_fn = entry.get("filename")
            filename = str(raw_fn) if raw_fn is not None else None
            raw_pid = entry.get("persona_id")
            persona_id = str(raw_pid).strip() if raw_pid is not None and str(raw_pid).strip() else default_persona_id
        else:
            raise HTTPException(
                status_code=400,
                detail=f"items[{idx - 1}] must be a string or an object with 'text'",
            )
        if not persona_id:
            raise HTTPException(
                status_code=400,
                detail=f"items[{idx - 1}]: persona_id is required",
            )
        normalized.append({
            "text": text,
            "filename": filename,
            "persona_id": persona_id,
        })
    return normalized


@router.post("/synthesize-batch")
async def synthesize_batch_zip(
    body: dict,
    manager: Any = Depends(_get_manager_dep()),
) -> Response:
    """複数テキストを順番に合成し、WAV 群と ``manifest.json`` をまとめた ZIP を返す。

    1 件でも合成に成功すれば ZIP を返し、失敗した要素は ``manifest.json`` に
    理由を記録する。全件失敗したときのみ HTTP エラーを返す。
    """
    default_persona_id = str(body.get("persona_id") or "").strip()
    items = _normalize_batch_items(body.get("items"), default_persona_id)
    apply_pronunciation = bool(body.get("apply_pronunciation", True))
    clean_text = bool(body.get("clean_text", True))

    pw = _playback_worker_module()
    if pw is None or not hasattr(pw, "synthesize_wav"):
        raise HTTPException(
            status_code=500,
            detail="playback_worker module unavailable; host tool loader not initialised",
        )

    loop = asyncio.get_event_loop()
    pad_width = max(3, len(str(len(items))))
    used_names: set[str] = {"manifest.json"}
    zip_buf = io.BytesIO()
    manifest_items: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    first_error: Exception | None = None

    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, item in enumerate(items, start=1):
            text = str(item["text"] or "")
            persona_id = str(item["persona_id"] or "")
            fallback_stem = f"{idx:0{pad_width}d}"
            raw_fn = item["filename"]
            # manifest.json と被らないように WAV 拡張子付きで一意化
            wav_name = _unique_wav_filename(
                _sanitize_wav_filename(raw_fn, fallback_stem),
                used_names,
            )

            if not text.strip():
                failed += 1
                manifest_items.append({
                    "index": idx,
                    "status": "error",
                    "persona_id": persona_id,
                    "filename": None,
                    "text": text,
                    "error": "empty text",
                })
                continue

            call = functools.partial(
                pw.synthesize_wav,
                persona_id=persona_id,
                text=text,
                apply_pronunciation=apply_pronunciation,
                clean_text=clean_text,
                timeout=_SYNTH_TIMEOUT_SEC,
            )
            try:
                result = await loop.run_in_executor(None, call)
            except Exception as exc:
                failed += 1
                if first_error is None:
                    first_error = exc
                manifest_items.append({
                    "index": idx,
                    "status": "error",
                    "persona_id": persona_id,
                    "filename": None,
                    "text": text,
                    "error": str(exc),
                })
                continue

            zf.writestr(wav_name, result.wav_bytes)
            succeeded += 1
            manifest_items.append({
                "index": idx,
                "status": "ok",
                "persona_id": persona_id,
                "filename": wav_name,
                "text": text,
                "applied_text": result.applied_text,
                "engine": result.engine_name,
                "duration_sec": round(result.duration_sec, 3),
                "sample_rate": result.sample_rate,
            })

        if succeeded == 0:
            if first_error is not None:
                raise _map_synthesis_exception(first_error, pw)
            raise HTTPException(status_code=400, detail="no voiceable items in batch")

        manifest = {
            "persona_id": default_persona_id or None,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total": len(items),
            "succeeded": succeeded,
            "failed": failed,
            "items": manifest_items,
        }
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem_base = default_persona_id or "voices"
    safe_stem = _sanitize_wav_filename(stem_base, "voices")[:-4]
    zip_name = f"{safe_stem}_voices_{ts}.zip"
    headers = {
        "Content-Disposition": _content_disposition("attachment", zip_name),
        "X-TTS-Batch-Total": str(len(items)),
        "X-TTS-Batch-Succeeded": str(succeeded),
        "X-TTS-Batch-Failed": str(failed),
        "Cache-Control": "no-store",
    }
    return Response(content=zip_buf.getvalue(), media_type="application/zip", headers=headers)

