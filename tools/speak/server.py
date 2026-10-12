"""音声合成サーバー: 共通の合成関数 (engine/synthesis_core) を HTTP で公開する。

別の PC の Voice TTS (remote_gpt_sovits.py) から合成の依頼を受け、ローカル合成と
同じエンジンのコードで声を作って返す。設計は docs/intent/synthesis_server.md。

エンジン専用の Python 環境の中で動く。SAIVerse 本体を import しない。Web の部品
(fastapi など) も使わず、標準ライブラリの HTTP サーバーで作る (Irodori-TTS の
環境には fastapi が無いため)。

起動は通常 serve.bat / serve.sh (serve_launcher.py) から。直接動かすなら:
``python server.py --engine gpt_sovits --port 8082 --data-dir <絶対パス>``

やり取り (従来のクライアントへの応答は変えず、足すだけ):

- ``GET  /health`` → ``{"ok", "model_loaded", "device", "engine", "protocol": 2}``
- ``GET  /profiles/{id}`` → ``{"exists": true, "ref_text"}`` / 404 ``{"exists": false}``
- ``PUT  /profiles/{id}`` multipart ``ref_audio`` + ``ref_text`` → ``{"ok": true}``
- ``POST /synthesize`` JSON ``{"profile_id", "text"}`` → 完成した WAV (従来どおり)。
  ``"params"`` を付けるとエンジンに渡し、``"stream": true`` を付けると応答本文で
  声の断片を子プロセスと同じフレーム形式 (subprocess_ipc) で順に返す。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import socket
import sys
import tempfile
import threading
import traceback
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

# .../tools/speak を import 経路に入れて engine パッケージを import する
# (subprocess_worker と同じ作法。本体や addon の tools 親は入れない)。
_SPEAK_DIR = Path(__file__).resolve().parent
if str(_SPEAK_DIR) not in sys.path:
    sys.path.insert(0, str(_SPEAK_DIR))

import numpy as np  # noqa: E402

from engine import subprocess_ipc as ipc  # noqa: E402
from engine.synthesis_core import (  # noqa: E402
    LOCAL_ENGINES,
    create_local_engine,
    iter_response_frames,
)

LOGGER = logging.getLogger("voice_tts.server")

PROTOCOL_VERSION = 2

_PACK_ROOT = _SPEAK_DIR.parent.parent
_CONFIG_PATH = _PACK_ROOT / "config" / "default.json"
_CONFIG_TEMPLATE_PATH = _PACK_ROOT / "config" / "default.json.template"

# この形の ID はそのままフォルダ名にする (従来のサーバーと同じ置き方で、
# 今の登録を引き継げる)。それ以外の ID (2026-08 より前に作られた日本語の
# ペルソナ ID など) は、ID のハッシュをフォルダ名にする。
_PLAIN_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_PROFILE_ID_CHARS = 128
_EXTENSION_RE = re.compile(r"^\.[a-z0-9]{1,8}$")
# 設定画面と同じ上限
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# 合成の依頼 (JSON) の上限。文章と調整値だけなので小さい
MAX_JSON_BYTES = 1024 * 1024
# GPT-SoVITS は長さがこの範囲から外れた参照音声で合成に失敗する
MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 10.0
# 一つの断片の書き出しの上限 (応答を読まない依頼元が順番を占め続けないため)
DEFAULT_WRITE_TIMEOUT = 30.0

# soundfile の形式名 → 拡張子 (元のファイル名に使える拡張子が無いとき)
_FORMAT_EXTENSIONS = {
    "WAV": ".wav", "WAVEX": ".wav", "FLAC": ".flac", "OGG": ".ogg", "MP3": ".mp3",
    "AIFF": ".aiff",
}


class RequestError(Exception):
    """断片を返す前の失敗。HTTP の状態コードと理由で返す。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def load_engine_config(engine_name: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Voice TTS の config/default.json の ``engines.<engine_name>`` を読む。

    default.json が無ければ default.json.template を読む (playback_worker と同じ
    読み元。ただしサーバーは default.json を作らない)。``override`` の項目が優先。
    """
    source = _CONFIG_PATH if _CONFIG_PATH.exists() else _CONFIG_TEMPLATE_PATH
    config: Dict[str, Any] = {}
    if source.exists():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
            config = dict((loaded.get("engines") or {}).get(engine_name) or {})
        except Exception as exc:
            LOGGER.error("failed to read voice-tts config %s: %s", source, exc)
    config.update(override or {})
    return config


def _float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 の声を 16bit PCM モノラルの WAV にする (従来のサーバーと同じ変換)。"""
    pcm = np.clip(np.rint(audio.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SynthesisService:
    """エンジン一つと、プロファイルの置き場所を持つ。HTTP は知らない。

    合成とプロファイルの登録は ``lock`` で一件ずつ処理する (GPU を取り合わない)。
    一件の合成は、始めから終わりまで順番を占める。
    """

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        data_dir: Path,
        *,
        engine_name: str = "gpt_sovits",
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        configured_device: Optional[str] = None,
    ):
        self.engine_factory = engine_factory
        self.engine_name = engine_name
        # GPT-SoVITS は合成の間プロセスの作業フォルダを移すので、置き場所は
        # 起動時に絶対パスにしておく
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.profiles_dir = self.data_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.write_timeout = write_timeout
        self.lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.engine: Any = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._configured_device = configured_device
        self._load_thread: Optional[threading.Thread] = None

    # -- model ---------------------------------------------------------

    def start_loading(self) -> None:
        """裏でモデルを読み込み始める。読み込み中の合成の依頼は 503 で返す。"""
        self._load_thread = threading.Thread(
            target=self._load, name="voice-tts-server-load", daemon=True,
        )
        self._load_thread.start()

    def _load(self) -> None:
        try:
            engine = self.engine_factory()
            # 読み込み口があるエンジン (GPT-SoVITS / Irodori-TTS) は今読む。
            # 無ければ最初の合成のときに読まれる
            load = getattr(type(engine), "_lazy_load", None)
            if callable(load):
                engine._lazy_load()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            with self._state_lock:
                self.load_error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("voice-tts server: model load failed: %s", exc)
            return
        with self._state_lock:
            self.engine = engine
            self.model_loaded = True
        LOGGER.info("voice-tts server: model loaded (engine=%s, device=%s)", self.engine_name, self.device())

    def device(self) -> str:
        engine = self.engine
        if engine is not None and getattr(engine, "device", None):
            return str(engine.device)
        return str(self._configured_device or "unknown")

    def health(self) -> Dict[str, Any]:
        with self._state_lock:
            body: Dict[str, Any] = {
                # 読み込みに失敗したサーバーは合成できないので ok を落とす
                "ok": self.load_error is None,
                "model_loaded": self.model_loaded,
                "device": self.device(),
                "engine": self.engine_name,
                "protocol": PROTOCOL_VERSION,
            }
            if self.load_error is not None:
                body["load_error"] = self.load_error
        return body

    def _require_engine(self) -> Any:
        with self._state_lock:
            if self.load_error is not None:
                raise RequestError(503, f"model failed to load: {self.load_error}")
            if not self.model_loaded:
                raise RequestError(503, "model is loading")
            return self.engine

    # -- profiles ------------------------------------------------------

    @staticmethod
    def validate_profile_id(profile_id: str) -> str:
        """ペルソナ ID をプロファイル ID として受け付けるか確かめる。

        英数字に限らない (日本語のペルソナ ID もある)。フォルダ名に素で使うのは
        英数字の ID だけなので (``profile_dir``)、ここでは区切り文字・制御文字・
        前後の空白・長さだけを断る。
        """
        if (
            not profile_id
            or len(profile_id) > MAX_PROFILE_ID_CHARS
            or profile_id != profile_id.strip()
            or profile_id in (".", "..")
            or any(ch in "/\\" or ord(ch) < 32 or ord(ch) == 127 for ch in profile_id)
        ):
            raise RequestError(
                400,
                f"profile_id must be 1-{MAX_PROFILE_ID_CHARS} characters without "
                "slashes, control characters or surrounding spaces",
            )
        return profile_id

    def profile_dir(self, profile_id: str) -> Path:
        """プロファイルのフォルダ。英数字の ID はそのまま、それ以外はハッシュの名前。"""
        if _PLAIN_PROFILE_ID_RE.match(profile_id):
            return self.profiles_dir / profile_id
        digest = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:32]
        return self.profiles_dir / f"id_{digest}"

    def read_profile(self, profile_id: str) -> Optional[Tuple[Path, str]]:
        """登録済みなら (参照音声の絶対パス, 書き起こし)。参照音声が無ければ未登録扱い。"""
        profile_dir = self.profile_dir(profile_id)
        try:
            data = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        ref_name = data.get("ref_audio")
        ref_text = data.get("ref_text")
        if not isinstance(ref_name, str) or not ref_name or not isinstance(ref_text, str):
            return None
        ref_path = profile_dir / Path(ref_name).name
        if not ref_path.is_file():
            return None
        return ref_path, ref_text

    def store_profile(self, profile_id: str, audio: bytes, filename: str, ref_text: str) -> None:
        """参照音声を確かめてから登録する。確かめられなければ 400。"""
        import soundfile as sf

        if not ref_text.strip():
            raise RequestError(400, "ref_text is empty")
        if not audio:
            raise RequestError(400, "ref_audio is empty")
        with self.lock:
            profile_dir = self.profile_dir(profile_id)
            original_ext = Path(filename or "").suffix.lower()
            # 確かめる前は profiles/ の直下に置く (断った登録でフォルダを残さない)
            fd, tmp_name = tempfile.mkstemp(dir=str(self.profiles_dir), prefix=".upload_", suffix=original_ext)
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(audio)
                try:
                    info = sf.info(str(tmp))
                except Exception as exc:
                    raise RequestError(400, f"ref_audio is not a readable audio file: {exc}") from exc
                if not info.samplerate or info.frames <= 0:
                    raise RequestError(400, "ref_audio contains no audio")
                seconds = info.frames / info.samplerate
                if not MIN_REF_SECONDS <= seconds <= MAX_REF_SECONDS:
                    raise RequestError(
                        400,
                        f"ref_audio is {seconds:.2f} s long; GPT-SoVITS needs a reference "
                        f"between {MIN_REF_SECONDS:g} and {MAX_REF_SECONDS:g} seconds",
                    )
                ext = original_ext if _EXTENSION_RE.match(original_ext) else _FORMAT_EXTENSIONS.get(
                    str(info.format).upper(), ".audio",
                )
                # 内容のハッシュを名前に入れる: エンジンは同じパスの参照音声を
                # 読み直さないので、差し替えたらパスが変わる必要がある
                ref_name = f"ref_{hashlib.sha256(audio).hexdigest()[:16]}{ext}"
                profile_dir.mkdir(parents=True, exist_ok=True)
                os.replace(tmp, profile_dir / ref_name)
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
            _atomic_write(
                profile_dir / "profile.json",
                json.dumps(
                    {"ref_audio": ref_name, "ref_text": ref_text, "profile_id": profile_id},
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            _atomic_write(profile_dir / "ref_text.txt", ref_text.encode("utf-8"))
            for old in profile_dir.glob("ref_*"):
                if old.name != ref_name and old.name != "ref_text.txt":
                    try:
                        old.unlink()
                    except OSError as exc:
                        LOGGER.warning("could not delete old reference %s: %s", old, exc)
        LOGGER.info("profile %s registered (%s, %.2f s)", profile_id, ref_name, seconds)

    # -- synthesis -----------------------------------------------------

    def synthesis_frames(self, profile_id: str, text: str, params: Dict[str, Any]) -> Iterator[bytes]:
        """順番を取ってから合成し、フレームを順に返すジェネレータ。

        順番 (``lock``) はジェネレータの中で取り、``finally`` で返す。呼び出し側が
        途中で閉じたら (依頼元の切断)、エンジンの合成も打ち切って次の依頼に
        順番を回す。最初のフレームの前の失敗は RequestError で知らせる。
        """
        engine = self._require_engine()
        with self.lock:
            profile = self.read_profile(profile_id)
            if profile is None:
                raise RequestError(404, f"profile {profile_id!r} is not registered")
            ref_path, ref_text = profile
            frames = iter_response_frames(engine, {
                "method": "stream",
                "text": text,
                "ref_audio": str(ref_path),
                "ref_text": ref_text,
                "params": params,
            })
            try:
                yield from frames
            finally:
                frames.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "SynthesisHTTPServer"
    _body_read = False  # この依頼の本文を読んだか (_dispatch が依頼ごとに戻す)

    def setup(self) -> None:
        # 一回の読み書きの上限。断片の書き出しが詰まった依頼元を切り離す
        self.timeout = self.server.service.write_timeout
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.info("%s - %s", self.address_string(), format % args)

    # -- helpers -------------------------------------------------------

    @property
    def service(self) -> SynthesisService:
        return self.server.service

    def _send_json(self, status: int, body: Dict[str, Any], *, close: bool = False) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, status: int, message: str, *, close: bool = False) -> None:
        self._send_json(status, {"ok": False, "error": message}, close=close)

    def _read_body(self, limit: int) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            # 本文の長さが分からないと次の依頼との境目が読めない
            raise RequestError(411, "Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError:
            raise RequestError(400, "invalid Content-Length") from None
        if length < 0:
            raise RequestError(400, "invalid Content-Length")
        if length > limit:
            raise RequestError(413, f"request body is larger than {limit} bytes")
        data = self.rfile.read(length)
        self._body_read = True
        if len(data) != length:
            raise RequestError(400, "request body ended early")
        return data

    def _drain_unread_body(self) -> bool:
        """読まずに答える依頼の本文を読み捨てる。読み捨てられなければ False。

        本文を残したまま答えると、残りが次の依頼として読まれ、依頼元はまだ送って
        いる途中で接続を切られる。読み捨てられないとき (長さ不明・上限超え) は、
        応答の後で接続を閉じる。
        """
        if self._body_read:
            return True
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return self.headers.get("Transfer-Encoding") is None
        try:
            remaining = int(raw_length)
        except ValueError:
            return False
        if remaining < 0 or remaining > MAX_UPLOAD_BYTES:
            return False
        while remaining:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                return False
            remaining -= len(chunk)
        self._body_read = True
        return True

    def _profile_id_from_path(self, path: str) -> str:
        return self.service.validate_profile_id(unquote(path[len("/profiles/"):]))

    def _not_found(self, _path: str) -> None:
        raise RequestError(404, "not found")

    def _dispatch(self, handler: Callable[[str], None], path: str) -> None:
        self._body_read = False
        try:
            handler(path)
        except RequestError as exc:
            # 本文を読み捨てられない (411 / 413 など) なら、この接続はもう使えない
            keep = exc.status not in (411, 413) and self._drain_unread_body()
            self._send_error_json(exc.status, exc.message, close=not keep)
        except (ConnectionError, socket.timeout) as exc:
            LOGGER.warning("client connection lost: %s", exc)
            self.close_connection = True
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            try:
                self._send_error_json(500, f"{type(exc).__name__}: {exc}", close=True)
            except Exception:
                self.close_connection = True

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/health":
            self._dispatch(lambda _p: self._send_json(200, self.service.health()), path)
        elif path.startswith("/profiles/"):
            self._dispatch(self._get_profile, path)
        else:
            self._dispatch(self._not_found, path)

    def do_PUT(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path.startswith("/profiles/"):
            self._dispatch(self._put_profile, path)
        else:
            self._dispatch(self._not_found, path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/synthesize":
            self._dispatch(self._synthesize, path)
        else:
            self._dispatch(self._not_found, path)

    def _get_profile(self, path: str) -> None:
        profile = self.service.read_profile(self._profile_id_from_path(path))
        if profile is None:
            self._send_json(404, {"exists": False})
            return
        self._send_json(200, {"exists": True, "ref_text": profile[1]})

    def _put_profile(self, path: str) -> None:
        profile_id = self._profile_id_from_path(path)
        content_type = self.headers.get("Content-Type", "")
        body = self._read_body(MAX_UPLOAD_BYTES)
        if not content_type.lower().startswith("multipart/form-data"):
            raise RequestError(400, "expected multipart/form-data with ref_audio and ref_text")
        fields = _parse_multipart(content_type, body)
        audio_field = fields.get("ref_audio")
        text_field = fields.get("ref_text")
        if audio_field is None:
            raise RequestError(400, "ref_audio is missing")
        if text_field is None:
            raise RequestError(400, "ref_text is missing")
        try:
            ref_text = text_field[1].decode("utf-8")
        except UnicodeDecodeError:
            raise RequestError(400, "ref_text must be UTF-8") from None
        self.service.store_profile(profile_id, audio_field[1], audio_field[0] or "", ref_text)
        self._send_json(200, {"ok": True})

    def _synthesize(self, _path: str) -> None:
        try:
            request = json.loads(self._read_body(MAX_JSON_BYTES).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise RequestError(400, "request body must be JSON") from None
        if not isinstance(request, dict):
            raise RequestError(400, "request body must be a JSON object")
        profile_id = self.service.validate_profile_id(str(request.get("profile_id") or ""))
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError(400, "text is empty")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise RequestError(400, "params must be a JSON object")
        stream = request.get("stream") is True

        frames = self.service.synthesis_frames(profile_id, text, params)
        try:
            # 最初のフレームまでは状態コードで失敗を返せる (順番待ち・404・
            # エンジンの即時の失敗)
            first = next(frames)
            if first[:1] == b"X":
                raise RequestError(500, ipc.decode_response(first)[1])
            if stream:
                self._send_stream(first, frames)
            else:
                self._send_wav(first, frames)
        finally:
            frames.close()

    def _send_wav(self, first: bytes, frames: Iterator[bytes]) -> None:
        """従来のクライアント向け: 全部の断片をつないで WAV を一括で返す。"""
        audio: List[np.ndarray] = []
        sample_rate: Optional[int] = None
        for payload in _chain(first, frames):
            kind, value = ipc.decode_response(payload)
            if kind in ("chunk", "result"):
                chunk, sr = value[0], value[1]
                if sample_rate is None:
                    sample_rate = int(sr)
                if chunk.size:
                    audio.append(chunk)
            elif kind == "error":
                raise RequestError(500, value)
            elif kind == "end":
                break
        if not audio or sample_rate is None:
            raise RequestError(500, "synthesis produced no audio")
        data = _float_to_wav_bytes(np.concatenate(audio), sample_rate)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_stream(self, first: bytes, frames: Iterator[bytes]) -> None:
        """声の断片を、子プロセスと同じ長さ付きフレームで chunked の本文に流す。

        書き出しに失敗したら (依頼元の切断・書き出しの時間切れ)、呼び出し側の
        finally がジェネレータを閉じ、合成を打ち切って順番を返す。
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for payload in _chain(first, frames):
                frame = ipc.frame_bytes(payload)
                self.wfile.write(b"%X\r\n%s\r\n" % (len(frame), frame))
            self.wfile.write(b"0\r\n\r\n")
        except (ConnectionError, socket.timeout, OSError) as exc:
            LOGGER.warning("stream to %s stopped: %s", self.address_string(), exc)
            self.close_connection = True


def _chain(first: bytes, rest: Iterator[bytes]) -> Iterator[bytes]:
    yield first
    yield from rest


def _parse_multipart(content_type: str, body: bytes) -> Dict[str, Tuple[Optional[str], bytes]]:
    """multipart/form-data を {名前: (ファイル名, 中身)} にする (標準ライブラリの email で)。"""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    try:
        message = BytesParser(policy=policy.HTTP).parsebytes(header + body)
    except Exception as exc:
        raise RequestError(400, f"invalid multipart body: {exc}") from exc
    if not message.is_multipart():
        raise RequestError(400, "invalid multipart body")
    fields: Dict[str, Tuple[Optional[str], bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        fields[str(name)] = (part.get_filename(), payload)
    return fields


class SynthesisHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # Windows の SO_REUSEADDR は使用中のポートにも bind できてしまうので切る
    # (同じポートで二つ目のサーバーを起動したら、黙って二重にせず失敗させる)
    allow_reuse_address = os.name != "nt"

    def __init__(self, address: Tuple[str, int], service: SynthesisService):
        self.service = service
        super().__init__(address, _Handler)


def create_server(
    engine_factory: Callable[[], Any],
    data_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8082,
    engine_name: str = "gpt_sovits",
    write_timeout: float = DEFAULT_WRITE_TIMEOUT,
    configured_device: Optional[str] = None,
    load: bool = True,
) -> SynthesisHTTPServer:
    """サーバーを作り、裏でモデルの読み込みを始める (``serve_forever`` は呼ばない)。

    テストは ``engine_factory`` に決まった音を返すエンジンを渡し、``port=0`` で
    空いているポートを使う。
    """
    service = SynthesisService(
        engine_factory, data_dir,
        engine_name=engine_name, write_timeout=write_timeout,
        configured_device=configured_device,
    )
    server = SynthesisHTTPServer((host, port), service)
    if load:
        service.start_loading()
    return server


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Voice TTS speech synthesis server")
    ap.add_argument("--engine", choices=LOCAL_ENGINES, default="gpt_sovits")
    ap.add_argument("--host", default="127.0.0.1",
                    help="listen address (default 127.0.0.1; use 0.0.0.0 for other PCs)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--data-dir", required=True, help="where registered profiles are stored")
    ap.add_argument("--config", default=None,
                    help="JSON object overriding config/default.json engines.<engine>")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    override: Dict[str, Any] = {}
    if args.config:
        try:
            override = json.loads(args.config)
        except ValueError as exc:
            print(f"--config is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(override, dict):
            print("--config must be a JSON object", file=sys.stderr)
            return 2
    config = load_engine_config(args.engine, override)
    server = create_server(
        lambda: create_local_engine(args.engine, config),
        Path(args.data_dir),
        host=args.host, port=args.port, engine_name=args.engine,
        configured_device=str(config.get("device", "cuda")),
    )
    LOGGER.info(
        "voice-tts server: listening on http://%s:%d (engine=%s, data=%s, python=%s)",
        args.host, server.server_address[1], args.engine, server.service.data_dir, sys.executable,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
