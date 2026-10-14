"""Background playback worker.

Single FIFO queue + dedicated thread. synthesize -> save wav -> play via
sounddevice on the backend machine. All operations are non-blocking from the
Tool caller's perspective.
"""
from __future__ import annotations

import atexit
import io
import json
import logging
import queue
import threading
import time
import uuid
import wave
from collections import OrderedDict
from concurrent.futures import Future
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .engine import TTSEngine, create_engine
from .profiles import get_profile
from .text_cleaner import clean_text_for_tts
from . import audio_stream

LOGGER = logging.getLogger(__name__)

_PACK_ROOT = Path(__file__).resolve().parent.parent.parent
# ユーザーローカル設定ファイル(.gitignore 対象、各環境固有の値を持つ)。
# 存在しない場合は .template から自動コピーされる(_load_config 内)。
_CONFIG_PATH = _PACK_ROOT / "config" / "default.json"
_CONFIG_TEMPLATE_PATH = _PACK_ROOT / "config" / "default.json.template"
_ADDON_NAME = "saiverse-voice-tts"

# ----- Chunk push 流量制御 (= subscriber 側 buffer overflow 抑制) ---
# 高水位 / 低水位パターン: 「subscriber に先行送信した秒数 (= lead)」 が
# 高水位を超えたら、 低水位まで戻るよう sleep する。 GPU 推論が realtime より
# 速いと chunk が無限に push されて subscriber 側 socket buffer 詰まり →
# client write timeout 連鎖死、 という観測 (session 20260515_131732 の
# stackchan_room:177 が 17.82s lead で死亡) への対策。
#
# 値の根拠: stack-chan (gateway → ESP32 forward + ESP32 内 buffer 含む) を
# 想定した保守値。 観測死亡値 17s に対して半分以下の高水位 + 安全マージン
# 込みの低水位。 web UI 等他の subscriber は通常もっと余裕があるので、
# 統一値で問題ない (= 厳しい subscriber 基準で全環境統一)。
_FLOW_LEAD_HIGH_SECONDS = 8.0
_FLOW_LEAD_LOW_SECONDS = 3.0
# subscriber が再生を開始するまでのラグ (= speak_hook の first chunk 待ち、
# gateway → ESP32 forward の startup、 ブラウザ HTML5 audio canplay 等)。
# 1 秒は推測値で、 実観測より小さく見積もる (= lead を大きめに出して安全側)。
_FLOW_SUBSCRIBER_STARTUP_ALPHA = 1.0


def _get_active_message_id() -> Optional[str]:
    try:
        from tools.context import get_active_message_id  # type: ignore
        mid = get_active_message_id()
        return str(mid) if mid is not None else None
    except Exception:
        return None


def _parse_output_device(value: Any) -> Optional[int]:
    """Resolve an output_device value (from addon UI or JSON config) to a
    ``sounddevice`` device index or None (= OS default).

    Accepted input shapes:
      - None / "" / "<default>"         → None
      - int                             → returned as-is
      - "3"                             → 3
      - "3: Speakers (Realtek)"         → 3   (UI dropdown format)

    Any unparsable value falls back to None so playback still reaches the
    OS default device instead of crashing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # Guard against bool being treated as int (True→1 is almost never intended here).
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s or s == "<default>":
        return None
    # Leading integer, optionally followed by ":" and label.
    head = s.split(":", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def _get_effective_params(persona_id: Optional[str]) -> Dict[str, Any]:
    """Return UI-driven addon params merged with pack-local defaults.

    Preference order:
      1. ``saiverse.addon_config.get_params`` (host UI-managed values, with
         persona-level overrides applied by the host)
      2. ``config/default.json`` of this pack (legacy / backward compat for
         SAIVerse builds without the addon framework)
      3. Hard-coded "enabled, everything on" fallback
    """
    cfg = _worker._load_config() if "_worker" in globals() else {}
    params: Dict[str, Any] = {
        "_enabled": True,
        "auto_speak": True,
        "server_side_playback": bool(cfg.get("server_side_playback", True)),
        "streaming": bool(cfg.get("streaming", True)),
    }
    try:
        from saiverse.addon_config import get_params  # type: ignore
        remote = get_params(_ADDON_NAME, persona_id=persona_id)
        if isinstance(remote, dict):
            params.update(remote)
    except Exception as exc:
        LOGGER.debug("addon_config.get_params unavailable: %s", exc)
    return params


def get_effective_params(persona_id: Optional[str]) -> Dict[str, Any]:
    """Public helper for the Tool to inspect effective addon settings."""
    return _get_effective_params(persona_id)


def _audio_path(message_id: str, version: Optional[str] = None) -> str:
    """フロント表示用の audio URL。

    ``version`` を付けるとクエリパラメータ ``?v=<version>`` を付与する。
    再生成 (同じ message_id でも新しい合成) のたびに version を変えれば、
    フロントの metadata 値が変化することを React 側で検知できるので、
    再生成中スピナーの完了判定や ``<audio>`` キャッシュバストに利用できる。
    FastAPI 側のルート照合は path だけ見るので、クエリ追加で挙動は変わらない。
    """
    base = f"/api/addon/{_ADDON_NAME}/audio/{message_id}"
    return f"{base}?v={version}" if version else base


def _audio_stream_url(message_id: str, version: Optional[str] = None) -> str:
    base = f"/api/addon/{_ADDON_NAME}/audio/{message_id}/stream"
    return f"{base}?v={version}" if version else base


def _notify_stream_ready(
    message_id: Optional[str],
    version: Optional[str] = None,
    pulse_id: Optional[str] = None,
) -> None:
    """Broadcast audio_ready at stream open time.

    ストリーミング推論で使用。合成完了を待たずに発火することで、クライアント側
    再生がレイテンシ少なく話し始められる。

    重要: ここでは ``audio_stream_url`` だけを通知する。``audio_path`` を
    早期に立てると、合成完了前に <audio> 要素が ``/audio/{msg}?v=NEW`` を
    取得して旧 wav 内容を新 URL でブラウザキャッシュしてしまう (URL は新版に
    更新されるが、当該 URL が指す ``audio_file`` メタデータは合成完了まで旧版
    のまま、というラグが原因)。
    ``audio_path`` の更新と SSE 通知は ``_notify_audio_ready`` (合成完了時)
    に集約することで、フロント側からは「URL が新版になった瞬間 = ファイルも
    新版」というアトミックな更新に見える。
    """
    if not message_id:
        LOGGER.warning(
            "notify_stream_ready skipped: message_id is None. "
            "Streaming client-side playback will not be triggered."
        )
        return
    stream_url = _audio_stream_url(message_id, version=version)
    try:
        from saiverse.addon_metadata import set_metadata  # type: ignore
        set_metadata(
            message_id=message_id,
            addon_name=_ADDON_NAME,
            key="audio_stream_url",
            value=stream_url,
        )
    except Exception as exc:
        LOGGER.warning("notify_stream_ready set_metadata failed for msg=%s: %s", message_id, exc)
    try:
        from saiverse.addon_events import emit_addon_event  # type: ignore
        # pulse_id を payload に載せると subscriber (= frontend / stackchan) が
        # 「同 pulse 連続発話 → queue 末尾」 「別 pulse 着信 → 旧再生中断 + 新
        # 再生」 の preempt 判定に使える (= Phase 2 設計)。 None ならフィールド
        # は省略して旧 subscriber と互換。
        event_data: Dict[str, Any] = {"audio_stream_url": stream_url}
        if pulse_id is not None:
            event_data["pulse_id"] = pulse_id
        emit_addon_event(
            addon=_ADDON_NAME,
            event="audio_ready",
            message_id=message_id,
            data=event_data,
        )
    except Exception as exc:
        LOGGER.warning("emit_addon_event(stream_ready) failed for msg=%s: %s", message_id, exc)


def _notify_audio_ready(
    message_id: Optional[str],
    wav_path: Path,
    event_name: str = "audio_ready",
    version: Optional[str] = None,
    pulse_id: Optional[str] = None,
) -> None:
    """Register wav metadata and broadcast a completion event.

    合成完了時に呼ばれる。ここで初めて ``audio_file`` (バックエンドが配信する
    実 wav パス) と ``audio_path`` (フロント向け URL、cache-bust 用 ?v= 付き)
    がセットになって更新される。アトミックなため、フロント側からは
    「audio_path URL が新版になった瞬間 = audio_file も新版」と見える。

    Args:
        event_name:
          - ``"audio_ready"`` (デフォルト): 非ストリーミング合成および
            初回完了通知用。frontend の ``auto_play_tts`` client_action が
            これを購読しているので発火すると自動再生が走る。
          - ``"audio_completed"``: ストリーミング合成の完了通知用。auto_play_tts
            は購読していないため二重発火しない。一方で
            ``useAddonEvents`` が任意のイベントで ``addonMetadata`` を
            マージするので、フロントの ``audio_path`` 値は更新される
            (= bubble の <audio> 要素が新 URL を取り直す)。
    """
    if not message_id:
        LOGGER.warning(
            "notify_audio_ready skipped: message_id is None. "
            "Bubble playback button will not be registered for this utterance."
        )
        return
    audio_path = _audio_path(message_id, version=version)
    meta_ok = False
    event_ok = False
    try:
        from saiverse.addon_metadata import set_metadata  # type: ignore
        # audio_path: フロントが <audio src=...> に使う URL
        set_metadata(
            message_id=message_id,
            addon_name=_ADDON_NAME,
            key="audio_path",
            value=audio_path,
        )
        # audio_file: バックエンド配信エンドポイントが実ファイルを開くためのローカルパス
        set_metadata(
            message_id=message_id,
            addon_name=_ADDON_NAME,
            key="audio_file",
            value=str(wav_path),
        )
        meta_ok = True
    except Exception as exc:
        LOGGER.warning("set_metadata failed for msg=%s: %s", message_id, exc)
    try:
        from saiverse.addon_events import emit_addon_event  # type: ignore
        # 同上 (= _notify_stream_ready 参照): pulse_id を載せておくと subscriber
        # が Phase 2 preempt 判定に使える。 audio_completed は streaming 経路の
        # 完了通知だが、 frontend 側で「現再生中の pulse_id」 を更新するために
        # も使えるので含めておく。
        event_data: Dict[str, Any] = {"audio_path": audio_path}
        if pulse_id is not None:
            event_data["pulse_id"] = pulse_id
        emit_addon_event(
            addon=_ADDON_NAME,
            event=event_name,
            message_id=message_id,
            data=event_data,
        )
        event_ok = True
    except Exception as exc:
        LOGGER.warning("emit_addon_event(%s) failed for msg=%s: %s", event_name, message_id, exc)
    LOGGER.debug(
        "notify_audio_ready: msg=%s metadata=%s event=%s name=%s",
        message_id, meta_ok, event_ok, event_name,
    )


def notify_no_audio(
    message_id: Optional[str],
    reason: str,
    pulse_id: Optional[str] = None,
) -> None:
    """「この吹き出しに音声は生まれない」を画面へ知らせる。

    画面の音声ボタンは ``addon.json`` の ``show_when: "metadata_exists"`` で、
    ``audio_path`` が立つまで回転する待ち表示を出している。音声が 1 件も作られ
    なかった吹き出しでは ``audio_path`` が永久に来ないので、待ち表示が残り続ける
    (SAIVerse 本体 ``AddonBubbleButtons.tsx``。時間切れの保険はあるが、5 分間
    回り続ける)。

    契約は ``unavailable_keys`` — **この吹き出しではもう立たないメタデータ鍵の
    一覧**。``audio_path`` に空文字や null を入れて済ませない理由が二つある:
    再生成ボタン (``regenerate``) は ``audio_path`` の値の変化を完了の合図に
    しているので偽の値を置くと合図が壊れること、そして再生ボタンが「値はある
    のに鳴らない」死んだボタンとして出てしまうこと。鍵の名前で言う形にすると、
    画面側の規則はアドオン非依存のまま (「そのボタンが待っている鍵が一覧に
    入っていたら待たない」) で済む。

    後から手動の再生成で本当に音声が作られたら ``audio_path`` が立ち、画面は
    実値のほうを優先するので、この知らせを取り消す必要はない。
    """
    if not message_id:
        return
    try:
        from saiverse.addon_metadata import set_metadata  # type: ignore
        set_metadata(
            message_id=message_id,
            addon_name=_ADDON_NAME,
            key="unavailable_keys",
            value=["audio_path", "audio_stream_url"],
        )
    except Exception as exc:
        LOGGER.warning("notify_no_audio set_metadata failed for msg=%s: %s", message_id, exc)
    try:
        from saiverse.addon_events import emit_addon_event  # type: ignore
        # event の data はそのままフロントの addonMetadata へマージされるので、
        # 再読込を待たずに待ち表示が解ける。reason はログと将来の診断用。
        event_data: Dict[str, Any] = {
            "unavailable_keys": ["audio_path", "audio_stream_url"],
            "audio_unavailable_reason": reason,
        }
        if pulse_id is not None:
            event_data["pulse_id"] = pulse_id
        emit_addon_event(
            addon=_ADDON_NAME,
            event="audio_unavailable",
            message_id=message_id,
            data=event_data,
        )
    except Exception as exc:
        LOGGER.warning("emit_addon_event(audio_unavailable) failed for msg=%s: %s", message_id, exc)
    LOGGER.debug("notify_no_audio: msg=%s reason=%s", message_id, reason)


def _saiverse_home() -> Path:
    import os
    env = os.getenv("SAIVERSE_HOME")
    if env:
        return Path(env)
    return Path.home() / ".saiverse"


#: 旧来の合成音声の保存先 (v0.5.x まで)。SAIVerse 本体の外 (スタンドアロン・
#: テスト) ではここを使い続ける。
LEGACY_OUT_DIR = _saiverse_home() / "user_data" / "voice" / "out"


def _out_dir() -> Path:
    """合成音声 (wav) の保存先 (作成はしない)。

    SAIVerse 本体の中ではアドオン永続データの規約の場所
    (``~/.saiverse/user_data/addon_data/saiverse-voice-tts/outputs/``)。
    旧来の ``user_data/voice/out/`` の中身は、本体の起動時の移行
    (``saiverse/addon_migrations.py``) が outputs/ へ移す。移行前に記録された
    旧来の絶対パスは、配信側 (api_routes._resolve_audio_file) が outputs/ で
    引き直す。import の時点ではフォルダを作らないよう、使うたびに解決する。
    """
    try:
        from saiverse.addon_paths import get_addon_data_dir  # type: ignore
    except ImportError:
        return LEGACY_OUT_DIR
    return get_addon_data_dir(_ADDON_NAME) / "outputs"


@dataclass
class _Job:
    job_id: str
    persona_id: Optional[str]
    text: str
    message_id: Optional[str] = None
    # ペルソナの 1 pulse 識別子。 SAIVerse 本体の persona_speak event に載って
    # きた値をそのまま下流 (= audio_ready event の subscriber) に伝搬させる。
    # subscriber 側 (frontend / stackchan) は pulse_id 比較で「同 pulse 連続
    # 発話 (= queue 末尾)」 vs 「別 pulse 着信 (= 旧再生中断 + 新再生)」 を
    # 切り分ける (= Phase 2 設計)。 詳細: docs/intent/voice_tts_playback_queue.md
    pulse_id: Optional[str] = None
    # Pipeline Streaming (= sea runtime 側の文区切り sub-speak emit) で同一
    # message_id に複数 sub-text を連続 enqueue するための識別子。 詳細:
    # docs/intent/voice_tts_pipeline_streaming.md
    #
    # ``sub_seq``: 同 message_id 内の連番 (1 から開始)。 None なら従来の
    # 「1 message = 1 job」 で動作 (= 互換)。
    # ``is_final``: この sub-text で当該 message の合成が完結する (= 後続が
    # 来ない、 audio_stream を close してよい合図)。 sub_seq=None の場合は
    # 必ず True 扱い。
    sub_seq: Optional[int] = None
    is_final: bool = True


class VoiceUnavailable(Exception):
    """``synthesize_piece`` がこのペルソナの声を作らない (待っても声は来ない)。

    アドオンが無効・そのペルソナの ``auto_speak`` が OFF・声プロファイルが無い・
    整形後に読み上げる文字が残らない、のいずれか。
    """


class NothingToVoice(VoiceUnavailable):
    """``synthesize_piece`` に渡した区切りが、整形 (``clean_text_for_tts``) か
    読み方の辞書 (``_apply_pronunciation``) を当てると読み上げる文字を残さない。

    もともと声の無い区切りなので、その発話の欠けとしては数えない
    (吹き出しの声は他の切れ端だけで付く)。
    """


class _EngineInitError(Exception):
    """``_TTSWorker._prepare_synthesis`` でエンジンの初期化に失敗した。"""

    def __init__(self, engine_name: str, cause: Exception) -> None:
        super().__init__(f"engine '{engine_name}' failed to initialize: {cause}")
        self.engine_name = engine_name
        self.cause = cause


@dataclass
class _SynthesisSetup:
    """``_process`` と ``_process_piece`` が共有する、合成前の解決結果。"""
    profile: Dict[str, Any]
    cfg: Dict[str, Any]
    effective: Dict[str, Any]
    engine_name: str
    engine: TTSEngine
    params: Dict[str, Any]
    ref_audio: Optional[str]
    ref_text: Optional[str]


@dataclass
class _PieceJob:
    """``synthesize_piece`` の依頼。通常の ``_Job`` と同じ ``_queue`` に並べ、
    同じワーカースレッドで処理する (エンジンを同時に二重に使わないため)。

    結果は ``future`` に ``(mono float32 の音声, sample_rate)`` で返す。
    依頼側は時間切れで ``future.cancel()`` するので、ワーカーは処理の前に
    ``set_running_or_notify_cancel()`` で取り消しを確かめる。
    """
    persona_id: str
    text: str
    message_id: str
    future: "Future[Tuple[np.ndarray, int]]"


@dataclass
class _MessageState:
    """同 message_id の sub-text 群 (= sub_seq=1, 2, ...) で共有する状態。

    Pipeline Streaming の voice-tts 側受け入れ機構 (Phase 2-α) で使う。
    sub_seq=1 (= 初回 sub-text) の合成中に audio_stream を open し、
    ``is_final=True`` の sub-text 完了時に close + wav 保存する。 中間 sub-text
    完了時には何も close せず ``collected`` に貯めて次の sub-text を待つ。

    ``sub_seq`` 機構を使わない 「1 message = 1 job」 経路 (= 既存互換) の場合
    でも _MessageState は使うが、 sub_seq=1 + is_final=True で 1 回開いて
    閉じる単純動作になる。
    """
    message_id: str
    pulse_id: Optional[str] = None
    sample_rate: Optional[int] = None
    first_chunk_at: Optional[float] = None
    # 流量制御 (= lead 計算) と wav 保存時の concatenate 用に push 済の audio
    # を sub-text 跨ぎで蓄積する
    collected: List[Any] = field(default_factory=list)
    # sounddevice OutputStream (server_side_playback=True 時に保持)。 同
    # message_id 内では同じ stream を持続使用、 is_final=True で stop+close
    sd_stream: Any = None
    # audio_stream registry に open_stream / open_pcm_stream を呼んだか
    audio_stream_opened: bool = False
    # 流量制御ログ + wav 保存ファイル名用に最初の sub_seq の job_id を覚える
    initial_job_id: Optional[str] = None
    # 累積発話時間 (= wav 保存時のデバッグ用)
    t_start: Optional[float] = None


class _TTSWorker:
    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[Union[_Job, _PieceJob]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # (engine name, remote transport) -> engine. transport is () for the
        # local engine and (url, timeout) for a remote GPT-SoVITS server.
        self._engines: Dict[tuple, TTSEngine] = {}
        # (engine name, remote url) -> current cache key, to close the old
        # engine when only that server's timeout setting changes.
        self._remote_engine_keys: Dict[tuple, tuple] = {}
        # Pipeline Streaming で同 message_id の sub-text 群が共有する状態。
        # _process は 1 worker thread から逐次 access するので競合しないが、
        # 「is_final=True の job が完了した時に state を drop」 を確実にする
        # ため明示的に管理する。
        self._message_states: Dict[str, _MessageState] = {}
        self._config: Dict[str, Any] = {}
        self._config_loaded = False
        self._lock = threading.Lock()

    def _load_config(self) -> Dict[str, Any]:
        if self._config_loaded:
            return self._config
        # ローカル版が無ければ .template から初回コピー (first-run materialization)。
        # ユーザーが local 版を編集して上流 pull で衝突しないようにするための仕組み。
        if not _CONFIG_PATH.exists() and _CONFIG_TEMPLATE_PATH.exists():
            try:
                _CONFIG_PATH.write_text(
                    _CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                LOGGER.info(
                    "Materialized %s from %s (first run).",
                    _CONFIG_PATH.name, _CONFIG_TEMPLATE_PATH.name,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to materialize %s from template: %s",
                    _CONFIG_PATH, exc,
                )
        # 読み込み元: ローカル版 → なければ template の内容を直接使う
        source = _CONFIG_PATH if _CONFIG_PATH.exists() else _CONFIG_TEMPLATE_PATH
        if source.exists():
            try:
                self._config = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                LOGGER.error("Failed to load voice-tts config from %s: %s", source, exc)
                self._config = {}
        self._config_loaded = True
        return self._config

    def _get_engine(self, name: str, effective: Optional[Dict[str, Any]] = None) -> TTSEngine:
        cfg = self._load_config()
        engine_cfg = dict((cfg.get("engines") or {}).get(name, {}))
        if name == "gpt_sovits":
            # UI の値を優先する。ただし UI の URL が空 (addon.json の既定) なら
            # config/default.json の engines.gpt_sovits の URL を残す。
            ui_url = str((effective or {}).get("gpt_sovits_remote_url") or "").strip()
            if ui_url:
                engine_cfg["gpt_sovits_remote_url"] = ui_url
            if effective is not None and "gpt_sovits_remote_timeout_sec" in effective:
                engine_cfg["gpt_sovits_remote_timeout_sec"] = effective["gpt_sovits_remote_timeout_sec"]
            override_url = str((effective or {}).get("gpt_sovits_remote_url_override") or "").strip()
            if override_url:
                engine_cfg["gpt_sovits_remote_url"] = override_url
        remote_url = str(engine_cfg.get("gpt_sovits_remote_url") or "").strip().rstrip("/")
        transport = (remote_url, engine_cfg.get("gpt_sovits_remote_timeout_sec", 15)) if remote_url else ()
        # Engines are cached per (engine, transport) so personas on the local
        # engine and personas on a remote server (or on different servers) can
        # alternate without killing the local child process and reloading the
        # model on every switch. Only a remote engine whose settings changed is
        # replaced.
        key = (name, transport)
        if key in self._engines:
            return self._engines[key]
        stale = self._remote_engine_keys.get((name, remote_url)) if remote_url else None
        if stale is not None and stale in self._engines:
            LOGGER.debug("voice-tts: replacing engine=%s after remote timeout changed", name)
            self._engines.pop(stale).close()
        engine = create_engine(name, engine_cfg)
        self._engines[key] = engine
        if remote_url:
            self._remote_engine_keys[(name, remote_url)] = key
        return engine

    def _save_wav(self, audio: np.ndarray, sample_rate: int, job_id: str) -> Path:
        import wave
        out_dir = _out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{job_id}.wav"
        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return path

    def _play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_device: Optional[int] = None,
    ) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            LOGGER.warning("sounddevice not installed; skipping playback.")
            return
        try:
            sd.play(audio, sample_rate, device=output_device, blocking=True)
        except Exception as exc:
            LOGGER.error("sounddevice playback failed: %s", exc)

    @staticmethod
    def _to_int16_bytes(audio_np: np.ndarray) -> bytes:
        clipped = np.clip(audio_np, -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    def _play_streaming(
        self,
        engine: TTSEngine,
        text: str,
        ref_audio: Optional[str],
        ref_text: Optional[str],
        params: Optional[Dict[str, Any]],
        job_id: str,
        message_id: Optional[str] = None,
        server_side_playback: bool = True,
        output_device: Optional[int] = None,
        pulse_id: Optional[str] = None,
        sub_seq: Optional[int] = None,
        is_final: bool = True,
    ) -> bool:
        """Synthesize and play chunk-by-chunk while saving the full wav.

        If ``message_id`` is provided, each chunk is also forwarded to the
        in-process ``audio_stream`` registry so that the HTTP endpoint
        ``/api/addon/saiverse-voice-tts/audio/<message_id>/stream`` can serve
        the same audio to remote clients via HTTP Chunked Transfer.

        Pipeline Streaming (Phase 2-α): when ``sub_seq`` is set, this method
        looks up / creates a ``_MessageState`` for ``message_id`` and uses
        it to keep ``audio_stream`` open across multiple sub-text jobs. The
        first sub-text (sub_seq=1) opens the streams and fires
        ``audio_ready``; intermediate sub-texts only push chunks; the final
        sub-text (``is_final=True``) closes the streams, saves the
        accumulated wav and fires ``audio_completed``. Old "1 message =
        1 job" behavior is preserved when ``sub_seq is None`` (or when
        sub_seq=1 + is_final=True, which is the same effective shape).

        Returns True on success, False if streaming fell through and caller
        should use the non-streaming fallback.
        """
        sd = None
        if server_side_playback:
            try:
                import sounddevice as sd  # type: ignore
            except ImportError:
                LOGGER.warning(
                    "sounddevice not installed; streaming to HTTP only "
                    "(server-side playback disabled)."
                )
                sd = None

        device = output_device

        # Pipeline Streaming: 同 message_id の sub-text 跨ぎで shared state を
        # 使う。 sub_seq=None / message_id=None なら ephemeral state で動かす
        # (= 従来の 「1 sub-text 完結」 と等価動作)。
        if message_id and sub_seq is not None:
            state = self._message_states.get(message_id)
            if state is None:
                state = _MessageState(message_id=message_id, pulse_id=pulse_id)
                self._message_states[message_id] = state
        else:
            state = _MessageState(
                message_id=message_id or "",
                pulse_id=pulse_id,
            )
        if state.t_start is None:
            state.t_start = time.time()
        if state.initial_job_id is None:
            state.initial_job_id = job_id
        # この文の声を一片でも流したかを見るため、始める前の断片数を覚える
        collected_before = len(state.collected)

        try:
            for chunk in engine.synthesize_stream(
                text=text, ref_audio=ref_audio, ref_text=ref_text, params=params,
            ):
                audio_np = chunk.audio
                if audio_np.ndim > 1:
                    audio_np = audio_np.reshape(-1)
                if audio_np.size == 0:
                    continue
                if state.sample_rate is None:
                    # 初回 chunk (= 同 message 内で最初に音が出る瞬間)。
                    # sd OutputStream + audio_stream / pcm_stream を開いて
                    # audio_ready を発火する。 sub_seq>=2 の sub-text は
                    # state.sample_rate が既に set されているのでここに来ない。
                    state.sample_rate = chunk.sample_rate
                    if sd is not None:
                        state.sd_stream = sd.OutputStream(
                            samplerate=state.sample_rate,
                            channels=1,
                            device=device,
                            dtype="float32",
                        )
                        state.sd_stream.start()
                    if message_id:
                        audio_stream.open_stream(message_id, state.sample_rate)
                        # PCM 経路も並行で open (Stack-chan 等の物理 vessel が
                        # MP3 decode を省いて直接 playRaw に流すための経路)
                        audio_stream.open_pcm_stream(
                            message_id, state.sample_rate, channels=1,
                        )
                        state.audio_stream_opened = True
                        # ストリーム開始直後に audio_ready を発火する。Route Handler
                        # 側で /stream エンドポイントは arrayBuffer バッファ展開を
                        # スキップして素通しするようになったため、クライアントは
                        # ここからチャンクを progressive に受け取って早期再生できる。
                        _notify_stream_ready(
                            message_id, version=state.initial_job_id, pulse_id=pulse_id,
                        )
                    state.first_chunk_at = time.time()
                    LOGGER.debug(
                        "TTS first chunk ready after %.2fs (job=%s, msg=%s, sub=%s)",
                        state.first_chunk_at - state.t_start, job_id, message_id, sub_seq,
                    )
                elif chunk.sample_rate != state.sample_rate:
                    # 同 message 内で sample_rate が変わるのは voice profile 切替等
                    # の異常系。 普通は起きないが起きたら警告だけ残して続行
                    # (= state.sample_rate のまま push、 音は崩れる可能性)。
                    LOGGER.warning(
                        "TTS sample_rate mismatch within message: msg=%s "
                        "expected=%d got=%d (sub=%s)",
                        message_id, state.sample_rate, chunk.sample_rate, sub_seq,
                    )
                # ----- Flow control (高水位/低水位 + α 補正) -----
                # 既に push 済 (= state.collected) の合計秒数と、 subscriber が
                # 再生できた推測秒数 (= elapsed - α) の差を 「lead」 として、
                # これが高水位を超えたら低水位まで戻すよう sleep する。 GPU
                # 推論が realtime より速いと chunk が無限に push されて
                # subscriber 側 buffer 詰まり → write timeout 連鎖死、 という
                # 観測 (session 20260515_131732 の stackchan_room:177 が 17.82s
                # lead で死亡) への対策。 詳細は intent doc 参照。
                #
                # state.collected は sub-text 跨ぎで累積するので、 sub_seq=2 の
                # 最初の chunk でも 「sub_seq=1 の合計」 が見えており lead 判定
                # が正しく機能する。
                if state.collected and state.first_chunk_at is not None and state.sample_rate:
                    total_sent_seconds = sum(
                        len(a) / state.sample_rate for a in state.collected
                    )
                    elapsed = time.time() - state.first_chunk_at
                    actual_played = elapsed - _FLOW_SUBSCRIBER_STARTUP_ALPHA
                    lead_seconds = total_sent_seconds - actual_played
                    if lead_seconds > _FLOW_LEAD_HIGH_SECONDS:
                        sleep_seconds = lead_seconds - _FLOW_LEAD_LOW_SECONDS
                        LOGGER.debug(
                            "flow control: lead=%.2fs > %.1fs, "
                            "sleep %.2fs to bring back to %.1fs "
                            "(job=%s, msg=%s, sub=%s)",
                            lead_seconds, _FLOW_LEAD_HIGH_SECONDS,
                            sleep_seconds, _FLOW_LEAD_LOW_SECONDS,
                            job_id, message_id, sub_seq,
                        )
                        time.sleep(sleep_seconds)

                if state.sd_stream is not None:
                    state.sd_stream.write(audio_np.astype(np.float32, copy=False))
                if state.audio_stream_opened:
                    pcm_bytes = self._to_int16_bytes(audio_np)
                    audio_stream.push_chunk(message_id, pcm_bytes)
                    # PCM 経路にも同じ bytes を broadcast
                    audio_stream.push_pcm_chunk(message_id, pcm_bytes)
                state.collected.append(audio_np)
        except Exception as exc:
            LOGGER.error(
                "Streaming synthesis/playback failed: %s (msg=%s, sub=%s)",
                exc, message_id, sub_seq,
            )
            if len(state.collected) == collected_before:
                # まだ一片も流していない: 呼び出し側が一括合成で作り直す
                self._teardown_message_state(state, message_id)
                return False
            # この文の前半はもう鳴っている。一文丸ごと作り直すと前半がもう一度
            # 鳴るので作り直さず、流した分で文を終えて、吹き出しの続き (次の文・
            # 完成 WAV の保存) は普段どおり進める。外部サーバーとの接続が途中で
            # 切れたときに起きやすい。
            LOGGER.warning(
                "Keeping the partial audio already played instead of re-synthesizing "
                "the whole sentence (msg=%s, sub=%s)",
                message_id, sub_seq,
            )

        # is_final=False (= 中間 sub-text) の場合は ここで return。 stream は
        # 開いたまま、 collected は累積したまま、 次の sub-text の
        # _play_streaming 呼び出しで再利用される。
        if not is_final:
            LOGGER.debug(
                "Streaming sub-text complete (intermediate): job=%s msg=%s sub=%s "
                "collected_chunks=%d",
                job_id, message_id, sub_seq, len(state.collected),
            )
            return True

        # is_final=True: 全 sub-text 合成終了。 stream を close + wav 保存 +
        # audio_completed event 発火 + state を drop。
        return self._finalize_message_state(state, message_id, pulse_id)

    def _teardown_message_state(
        self, state: _MessageState, message_id: Optional[str]
    ) -> None:
        """例外発生時に state を完全 close + drop してリソースリークを防ぐ。

        is_final 待たずに発火するので、 audio_stream は 「合成が途中で死んだ」
        状態で close される (= subscriber 側は突然 stream 終わる)。
        """
        if state.sd_stream is not None:
            try:
                state.sd_stream.stop()
                state.sd_stream.close()
            except Exception:
                pass
            state.sd_stream = None
        if state.audio_stream_opened and message_id:
            audio_stream.close_stream(message_id)
            audio_stream.close_pcm_stream(message_id)
            state.audio_stream_opened = False
        if message_id and state.message_id == message_id:
            self._message_states.pop(message_id, None)

    def _finalize_message_state(
        self,
        state: _MessageState,
        message_id: Optional[str],
        pulse_id: Optional[str],
    ) -> bool:
        """is_final=True の sub-text 完了時に呼ぶ。 stream を綺麗に閉じ、
        累積 wav を保存して audio_completed を emit、 state を drop する。
        """
        # sd OutputStream を閉じる
        if state.sd_stream is not None:
            try:
                state.sd_stream.stop()
                state.sd_stream.close()
            except Exception:
                pass
            state.sd_stream = None

        # audio_stream registry を閉じる
        if state.audio_stream_opened and message_id:
            audio_stream.close_stream(message_id)
            audio_stream.close_pcm_stream(message_id)
            state.audio_stream_opened = False

        # state を dict から外す (= 同 message_id の次の発話用に空ける)
        if message_id:
            self._message_states.pop(message_id, None)

        if not state.collected or state.sample_rate is None:
            LOGGER.warning(
                "Streaming produced no audio for msg=%s (initial job=%s)",
                message_id, state.initial_job_id,
            )
            # state はあったのに一片も音が溜まらなかった回 (全 sub-text の合成が
            # 失敗した等)。ここも audio_path が立たないまま終わるので、上の
            # 「state が無い」回と同じ知らせを出して画面の待ち表示を解く。
            notify_no_audio(
                message_id,
                "synthesis produced no audio for this message",
                pulse_id=pulse_id,
            )
            return False

        try:
            full = np.concatenate(state.collected)
            t_total = (
                time.time() - state.t_start if state.t_start is not None else 0.0
            )
            wav_path = self._save_wav(full, state.sample_rate, state.initial_job_id or "")
            LOGGER.debug(
                "TTS streamed wav saved: %s (%d ms, total %.2fs, msg=%s)",
                wav_path, int(len(full) / state.sample_rate * 1000),
                t_total, message_id,
            )
            # ストリーミング経路は stream_ready で audio_ready 発火済み。
            # ここでは完成 wav の metadata を確定させ、別イベント名で通知して
            # auto_play_tts の二重発火を避けつつフロントの addonMetadata を更新する。
            _notify_audio_ready(
                message_id, wav_path,
                event_name="audio_completed",
                version=state.initial_job_id,
                pulse_id=pulse_id,
            )
        except Exception as exc:
            LOGGER.warning("Failed to save streamed wav: %s (msg=%s)", exc, message_id)

        return True

    def _gc_old_files(self) -> None:
        cfg = self._load_config()
        hours = float(cfg.get("gc_hours", 24))
        out_dir = _out_dir()
        if hours <= 0 or not out_dir.exists():
            return
        cutoff = time.time() - hours * 3600
        for path in out_dir.glob("*.wav"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    def _prepare_synthesis(self, persona_id: Optional[str]) -> _SynthesisSetup:
        """声プロファイル・エンジン・合成パラメータ・参照音声を解決する。

        ``_process`` (画面向けの通常合成) と ``_process_piece`` (仮想身体向けの
        一切れの合成) が同じ解決を使うための共通部分。

        Raises:
            VoiceUnavailable: 声プロファイルが無い (``_default`` も無い)。
            _EngineInitError: エンジンの初期化に失敗した。
        """
        profile = get_profile(persona_id)
        if profile is None:
            LOGGER.info(
                "No voice profile for persona_id=%s (and no _default); skipping TTS.",
                persona_id,
            )
            raise VoiceUnavailable("no voice profile")

        cfg = self._load_config()
        engine_name = profile.get("engine") or cfg.get("default_engine", "gpt_sovits")
        effective = _get_effective_params(persona_id)
        try:
            engine = self._get_engine(engine_name, effective)
        except Exception as exc:
            LOGGER.error("Failed to initialize engine '%s': %s", engine_name, exc)
            raise _EngineInitError(engine_name, exc) from exc

        synthesis_params = dict(profile.get("params") or {})
        if persona_id:
            synthesis_params["persona_id"] = persona_id
        ref_audio = profile.get("ref_audio")
        ref_text = profile.get("ref_text")
        if profile.get("is_default_voice") and getattr(engine, "stores_reference_by_persona", False):
            # 見本の声をこのペルソナの ID で外部サーバーに登録しない (共有サーバー
            # 上の本人の声を上書きしうる)。サーバー登録済みのプロファイルで合成する。
            LOGGER.debug(
                "persona=%s has only the sample voice; using the server's profile as is",
                persona_id,
            )
            ref_audio = ref_text = None
        return _SynthesisSetup(
            profile=profile,
            cfg=cfg,
            effective=effective,
            engine_name=engine_name,
            engine=engine,
            params=synthesis_params,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    @staticmethod
    def _apply_pronunciation(
        text: str, profile: Dict[str, Any], persona_id: Optional[str]
    ) -> str:
        # ユーザー読み方辞書の適用 (TTS engine に渡す直前に文字列置換)。
        # global 辞書 + ペルソナ別オーバーライド (registry の pronunciation_dict)
        # が両方ある場合は persona の方が先に適用される。
        from . import pronunciation_dict as _pd
        persona_dict = profile.get("pronunciation_dict")
        tts_text = _pd.apply(text, persona_dict=persona_dict)
        if tts_text != text:
            LOGGER.debug(
                "pronunciation_dict applied: %d -> %d chars (persona=%s)",
                len(text), len(tts_text), persona_id,
            )
        return tts_text

    def _process_piece(self, job: _PieceJob) -> None:
        """``synthesize_piece`` の一切れを合成して ``job.future`` へ返す。

        再生・audio_stream への配信・画面への通知・ファイル保存はしない
        (吹き出しの声は ``finish_message`` がまとめて作る)。
        """
        if not job.future.set_running_or_notify_cancel():
            LOGGER.debug("voice piece cancelled before synthesis (msg=%s)", job.message_id)
            return
        try:
            # 依頼の受付時 (synthesize_piece) と実際に始める今の間に、アドオンの
            # 無効化や auto_speak の OFF が挟まりうる (前の合成を待っている間)。
            # 始める時点の設定で改めて確かめ、エンジンの初期化より前に断る。
            effective = _get_effective_params(job.persona_id)
            if not effective.get("_enabled", True):
                raise VoiceUnavailable("voice-tts addon is disabled")
            if not effective.get("auto_speak", True):
                raise VoiceUnavailable("auto_speak is off for this persona")
            setup = self._prepare_synthesis(job.persona_id)
            cleaned = clean_text_for_tts(job.text)
            tts_text = self._apply_pronunciation(cleaned, setup.profile, job.persona_id)
            if not tts_text.strip():
                # 読み方の辞書で消える区切りも、synthesize_piece の入口の
                # NothingToVoice と同じく、もともと声の無い区切りとして扱う
                raise NothingToVoice("no voiceable text in this piece")
            result = setup.engine.synthesize(
                text=tts_text,
                ref_audio=setup.ref_audio,
                ref_text=setup.ref_text,
                params=setup.params,
            )
            audio = _mono_float32(result.audio)
            if audio.size == 0:
                raise RuntimeError("synthesis produced no audio")
            sample_rate = int(result.sample_rate)
            if sample_rate <= 0:
                raise RuntimeError("synthesis returned an invalid sample rate")
        except NothingToVoice as exc:
            LOGGER.debug(
                "voice piece has nothing to voice after the pronunciation dict "
                "(persona=%s msg=%s)", job.persona_id, job.message_id,
            )
            job.future.set_exception(exc)
            return
        except Exception as exc:
            LOGGER.warning(
                "voice piece synthesis failed (persona=%s msg=%s): %s",
                job.persona_id, job.message_id, type(exc).__name__,
            )
            job.future.set_exception(exc)
            return
        job.future.set_result((audio, sample_rate))

    def _process(self, job: _Job) -> None:
        # Pipeline Streaming finalize-only signal: text="" + is_final=True
        # の job は 「合成は不要、 既存 _MessageState を閉じて wav 保存だけ」
        # という SAIVerse 本体側の依頼。 既存 state があれば finalize する。
        # state が無いのは、この message に声にする文が一度も来なかった回
        # (本当に空・記号だけの吹き出し、または合成が失敗して state を捨てた
        # 回)。黙って抜けると画面の音声ボタンが audio_path を待ち続けて回転し
        # 続けるので、「この吹き出しに音声は無い」を知らせてから抜ける。
        if not job.text and job.is_final and job.message_id:
            state = self._message_states.get(job.message_id)
            if state is None:
                LOGGER.debug(
                    "finalize-only job for unknown msg=%s; "
                    "telling the UI this bubble has no audio",
                    job.message_id,
                )
                notify_no_audio(
                    job.message_id,
                    "no voiceable text was streamed for this message",
                    pulse_id=job.pulse_id,
                )
                return
            self._finalize_message_state(state, job.message_id, job.pulse_id)
            return

        try:
            setup = self._prepare_synthesis(job.persona_id)
        except VoiceUnavailable:
            notify_no_audio(job.message_id, "no voice profile", pulse_id=job.pulse_id)
            return
        except _EngineInitError as exc:
            notify_no_audio(
                job.message_id,
                f"engine '{exc.engine_name}' failed to initialize: {exc.cause}",
                pulse_id=job.pulse_id,
            )
            return
        profile = setup.profile
        cfg = setup.cfg
        engine_name = setup.engine_name
        effective = setup.effective
        engine = setup.engine
        synthesis_params = setup.params
        ref_audio = setup.ref_audio
        ref_text = setup.ref_text
        LOGGER.debug(
            "effective addon params for persona=%s: streaming=%s server_side=%s enabled=%s",
            job.persona_id,
            effective.get("streaming"),
            effective.get("server_side_playback"),
            effective.get("_enabled"),
        )
        use_streaming = bool(effective.get("streaming", True)) and getattr(
            engine, "supports_streaming", False
        )
        server_side_playback = bool(effective.get("server_side_playback", True))

        # UI で指定されたデバイス（"<default>" / "3: Realtek..." 等）を優先し、
        # 未指定なら config/default.json の output_device にフォールバックする。
        ui_device = effective.get("output_device")
        resolved_device = _parse_output_device(ui_device)
        if resolved_device is None:
            resolved_device = _parse_output_device(cfg.get("output_device"))

        tts_text = self._apply_pronunciation(job.text, profile, job.persona_id)

        if use_streaming:
            ok = self._play_streaming(
                engine=engine,
                text=tts_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                params=synthesis_params,
                job_id=job.job_id,
                message_id=job.message_id,
                server_side_playback=server_side_playback,
                output_device=resolved_device,
                pulse_id=job.pulse_id,
                sub_seq=job.sub_seq,
                is_final=job.is_final,
            )
            if ok:
                return
            LOGGER.info("Streaming failed; falling back to non-streaming synthesis.")

        try:
            result = engine.synthesize(
                text=tts_text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                params=synthesis_params,
            )
        except Exception as exc:
            LOGGER.error("TTS synthesis failed (engine=%s): %s", engine_name, exc)
            # 導入時に選ばなかったエンジンを指定したときもここに来る (専用環境が
            # 無く、子プロセス側の _prepare_sys_path が失敗する)。待ち表示を
            # 回し続けない。
            notify_no_audio(
                job.message_id,
                f"synthesis failed (engine={engine_name}): {exc}",
                pulse_id=job.pulse_id,
            )
            return

        wav_path: Optional[Path] = None
        try:
            wav_path = self._save_wav(result.audio, result.sample_rate, job.job_id)
            LOGGER.debug("TTS wav saved: %s (%d ms)", wav_path, result.duration_ms)
        except Exception as exc:
            LOGGER.warning("Failed to save wav: %s", exc)

        if wav_path is not None:
            _notify_audio_ready(
                job.message_id, wav_path, version=job.job_id,
                pulse_id=job.pulse_id,
            )

        if server_side_playback:
            self._play(result.audio, result.sample_rate, output_device=resolved_device)

    def _run(self) -> None:
        last_gc = 0.0
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                now = time.time()
                if now - last_gc > 3600:
                    self._gc_old_files()
                    last_gc = now
                continue
            if job is None:
                break
            try:
                if isinstance(job, _PieceJob):
                    self._process_piece(job)
                else:
                    self._process(job)
            except Exception:
                LOGGER.exception("Unhandled error in TTS worker")
            finally:
                self._queue.task_done()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="voice-tts-worker", daemon=True
            )
            self._thread.start()
            atexit.register(self.shutdown)

    def enqueue(
        self,
        text: str,
        persona_id: Optional[str],
        message_id: Optional[str] = None,
        pulse_id: Optional[str] = None,
        sub_seq: Optional[int] = None,
        is_final: bool = True,
    ) -> str:
        self.start()
        job_id = uuid.uuid4().hex
        # 通常経路 (speak_as_persona ツール) では contextvars から message_id を
        # 取得する。再生成 API のように context を持たない呼び出しは引数で
        # 明示的に渡せるようにする。
        captured_msg_id = message_id or _get_active_message_id()
        LOGGER.debug(
            "enqueue: job=%s persona=%s message_id=%s pulse_id=%s sub_seq=%s is_final=%s",
            job_id, persona_id, captured_msg_id, pulse_id, sub_seq, is_final,
        )
        self._queue.put(
            _Job(
                job_id=job_id,
                persona_id=persona_id,
                text=text,
                message_id=captured_msg_id,
                pulse_id=pulse_id,
                sub_seq=sub_seq,
                is_final=is_final,
            )
        )
        return job_id

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


_worker = _TTSWorker()


def enqueue_tts(
    text: str,
    persona_id: Optional[str],
    message_id: Optional[str] = None,
    pulse_id: Optional[str] = None,
    sub_seq: Optional[int] = None,
    is_final: bool = True,
) -> str:
    """TTS ジョブを enqueue する。

    Args:
        text: 合成テキスト
        persona_id: 発話ペルソナ
        message_id: バブル紐付け用 ID。None なら呼び出し時の contextvars から取得
            (通常の speak_as_persona ツール経路)。再生成 API のように context が
            無い経路は明示的に渡す。
        pulse_id: ペルソナ pulse 識別子。 audio_ready event payload に載せて
            subscriber 側 (frontend / stackchan) の同 pulse / 別 pulse 判定に
            使う (= Phase 2)。 None なら subscriber 側は preempt 判定不能で
            FIFO wait に倒す (= Phase 1 互換動作)。
        sub_seq: Pipeline Streaming (Phase 2-α) の sub-text 連番 (1 から)。
            同 message_id に複数回 enqueue する経路で使う。 None なら従来の
            「1 message = 1 job」 動作 (= 互換)。
        is_final: この sub-text で当該 message の合成が完結する合図。
            True なら audio_stream を close + wav 保存 + audio_completed 発火。
            False なら stream open 持続で次の sub-text を待つ。 sub_seq=None の
            場合は必ず True 扱い (= 1 sub-text 完結)。
    """
    return _worker.enqueue(
        text, persona_id,
        message_id=message_id, pulse_id=pulse_id,
        sub_seq=sub_seq, is_final=is_final,
    )


# ---------------------------------------------------------------------------
# 仮想身体 (Godot Vessel アドオン) への声の受け渡し
#
# ペルソナが仮想身体の部屋にいて Vessel が接続中の間、声は仮想身体のアバター
# からだけ鳴らす。設計: SAIVerse 本体 docs/intent/virtual_embodiment_godot.md
# の「声は voice-tts から一切れずつ受け取る」。
#
# - Godot アドオンは ``register_voice_claim`` で判定関数を登録する。
#   ``speak_hook.on_persona_speak`` と Godot アドオンは断片ごとに
#   ``claim_message_voice`` を呼ぶ。判定は message_id ごとに最初の一回で決まり、
#   真ならその発話を通常の合成 (ブラウザ・サーバーのスピーカー) に流さない。
# - Godot アドオンは全身エモート記法で区切った文を一つずつ
#   ``synthesize_piece`` に渡し、WAV を受け取る。合成は既存のワーカースレッドで
#   行う。
# - 最後の区切りを配送したら ``finish_message`` を呼ぶ。voice-tts は切れ端を
#   順につないで吹き出しの「音声を再生」に付ける (自動再生しない
#   ``audio_completed``)。一切れでも欠けたら ``notify_no_audio``。
#
# message_id ごとの記録 (``_VesselMessage``) は ``_vessel_messages`` 一つにだけ
# 持ち、その ``state`` が次のように移る:
#
#   (記録なし) --claim_message_voice が偽--> "unclaimed"  通常の合成が吹き出しを持つ
#   (記録なし) --claim_message_voice が真--> "open"       finish_message 側が吹き出しを持つ
#   "open" --finish_message / 期限切れ / 上限での追い出し--> "closed"
#
# - 記録を作るのは ``claim_message_voice`` だけ。一度決まった判定 ("unclaimed"
#   か、それ以外か) は変わらない。
# - 吹き出しへの結末の知らせ (``audio_completed`` か ``notify_no_audio``) を
#   出すのは、"open" から "closed" へ移る一回だけ。
# - "unclaimed" と "closed" は終わった状態。その後に来た切れ端は声を返すだけで、
#   記録にも吹き出しにも触らない。
# ---------------------------------------------------------------------------

#: "open" の発話を、触られない (断片・切れ端・締めのどれも来ない) まま持ち続ける
#: 上限秒数。過ぎたら "closed" にし、``notify_no_audio`` を出す。
#: ただし ``register_voice_claim`` の ``holds(message_id)`` のどれかが真を返す
#: 発話 (アドオンがまだ順番待ちか処理中で、いずれ ``finish_message`` を呼ぶ) は
#: 閉じずに触り直す。アドオン自身の合成待ちが長くても、締めを待ち続けられる。
#: なのでこの期限は、答えなくなったアドオン (プロセスが落ちた・アドオンが外れた、
#: または ``holds`` を登録していない) のための最後の網でしかない。
#: 画面の待ち表示はフロントが 300 秒で自分から消すので、長くしても待ち表示が
#: 残り続けることはない。
_VESSEL_MESSAGE_TTL_SEC = 3600.0
#: 期限切れを見回るスレッド (``_run_vessel_sweeper``) の起床間隔 (秒)。
_VESSEL_SWEEP_INTERVAL_SEC = 30.0
#: ``_vessel_messages`` の件数の目安。超えたら、終わってから
#: ``_VESSEL_RETAIN_TERMINAL_SEC`` 以上たった記録を捨てる
#: (``_evict_over_capacity_locked``)。"open" の記録は件数では捨てない。
_VESSEL_MESSAGE_MAX = 4096
#: 終わった記録 ("unclaimed" / "closed") を残しておく秒数 (1 日)。
_VESSEL_RETAIN_TERMINAL_SEC = 86400.0

_VESSEL_UNCLAIMED = "unclaimed"
_VESSEL_OPEN = "open"
_VESSEL_CLOSED = "closed"


@dataclass
class _VesselMessage:
    """一つの発話の、仮想身体への声の受け渡しの記録。

    ``state`` は ``_VESSEL_UNCLAIMED`` / ``_VESSEL_OPEN`` / ``_VESSEL_CLOSED``
    (この節の冒頭の図を参照)。``pieces`` の各要素は依頼順の枠で、合成が成功
    したら ``(mono float32 の音声, sample_rate)`` が入る。失敗・時間切れの枠は
    ``None`` のまま残り、``broken`` が立つ。ワーカーが ``NothingToVoice`` を
    返した枠 (読み方の辞書を当てると読み上げる文字が残らない) は ``skipped`` に
    番号を入れ、``finish_message`` はその枠を除いてつなぐ。"closed" になったら
    ``pieces`` は空にする (音声を持ち続けない)。
    """
    state: str
    created_at: float
    touched_at: float
    pulse_id: Optional[str] = None
    pieces: List[Optional[Tuple[np.ndarray, int]]] = field(default_factory=list)
    skipped: Set[int] = field(default_factory=set)
    broken: bool = False


@dataclass
class _VesselSweeper:
    """"open" の記録の期限切れを、API の呼び出しが無くても見回るスレッド。"""
    thread: threading.Thread
    stop: threading.Event


# _voice_claims・_vessel_messages・_vessel_sweeper を守る。判定関数
# (_is_voice_claimed)・``holds`` (_vessel_held_by_addon) と吹き出しへの知らせ
# (notify_no_audio / _notify_audio_ready) は、このロックを持ったまま呼ばない。
_vessel_lock = threading.Lock()
# claim_message_voice の判定を一件ずつ順に行う。同じ message_id の最初の断片が
# 同時に来ても、判定関数は一回だけ走る。取る順は _vessel_decision_lock →
# _vessel_lock で、逆順には取らない。
_vessel_decision_lock = threading.Lock()
# 登録された (判定関数, holds) の組。holds は登録されていなければ None。
_voice_claims: List[Tuple[Callable[[str, str], bool], Optional[Callable[[str], bool]]]] = []
# 並びは触った順 (先頭が最も古い)。上限超過時は先頭から捨てる。
_vessel_messages: "OrderedDict[str, _VesselMessage]" = OrderedDict()
# 動いている見回りスレッド。"open" の記録が無くなると自分で抜けて None に戻し、
# 次に記録が "open" になったときに改めて起こす (_ensure_vessel_sweeper_locked)。
_vessel_sweeper: Optional[_VesselSweeper] = None

# 吹き出しへ notify_no_audio で知らせる、閉じた発話 (message_id, pulse_id,
# 切れ端の数, 理由)
_ClosedNotice = Tuple[str, Optional[str], int, str]


def _mono_float32(audio: Any) -> np.ndarray:
    """エンジンの戻り値 (float32 または整数 PCM) を一次元の float32 にする。"""
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max + 1)
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        # _play_streaming と同じ扱い (エンジンは mono を返す前提)
        arr = arr.reshape(-1)
    return arr


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """mono float32 の音声を 16-bit PCM の WAV バイト列にする。"""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(_TTSWorker._to_int16_bytes(audio))
    return out.getvalue()


def register_voice_claim(
    fn: Callable[[str, str], bool],
    holds: Optional[Callable[[str], bool]] = None,
) -> None:
    """発話の声を仮想身体が引き受けるかの判定関数を登録する。

    ``fn(persona_id, building_id)`` が真を返した発話は、voice-tts の通常の
    合成 (ブラウザ・サーバーのスピーカー) に流さない。

    ``holds(message_id)`` は「このアドオンはこの発話をまだ順番待ちか処理中で
    持っていて、いずれ ``finish_message`` を呼ぶ」に答える。真を返す発話は
    ``_VESSEL_MESSAGE_TTL_SEC`` を過ぎても期限切れで閉じない
    (``_expire_vessel_messages``)。

    同じ関数オブジェクトの二度目の登録は何もしない (``holds`` も差し替えない)。
    """
    with _vessel_lock:
        if not any(existing is fn for existing, _ in _voice_claims):
            _voice_claims.append((fn, holds))


def unregister_voice_claim(fn: Callable[[str, str], bool]) -> None:
    """``register_voice_claim`` で登録した判定関数を、その ``holds`` ごと外す。

    未登録なら何もしない。
    """
    with _vessel_lock:
        _voice_claims[:] = [pair for pair in _voice_claims if pair[0] is not fn]


def _is_voice_claimed(persona_id: str, building_id: str) -> bool:
    """登録された判定関数のどれかが真なら真。例外は記録して偽として扱う。

    判定関数は他のアドオンのコードなので、``_vessel_lock`` を持たずに呼ぶ。
    """
    with _vessel_lock:
        claims = [fn for fn, _ in _voice_claims]
    for fn in claims:
        try:
            if fn(persona_id, building_id):
                return True
        except Exception:
            LOGGER.exception(
                "voice claim check failed (persona=%s building=%s)",
                persona_id, building_id,
            )
    return False


def _touch_locked(message_id: str, entry: _VesselMessage, now: float) -> None:
    """記録を触った時刻を更新し、上限超過時に捨てる順の末尾へ回す。

    ``entry`` が既に ``_vessel_messages`` から追い出されていれば並びは触らない。
    ``_vessel_lock`` を持った状態で呼ぶ。
    """
    entry.touched_at = now
    if _vessel_messages.get(message_id) is entry:
        _vessel_messages.move_to_end(message_id)


def _close_locked(message_id: str, entry: _VesselMessage, now: float) -> None:
    """記録を "closed" にする。``_vessel_lock`` を持った状態で呼ぶ。"""
    entry.state = _VESSEL_CLOSED
    entry.pieces = []
    _touch_locked(message_id, entry, now)


def _notify_closed(notices: List[_ClosedNotice]) -> None:
    """``finish_message`` を待たずに閉じた発話の吹き出しへ、声が無いことを知らせる。

    ``_vessel_lock`` を持たずに呼ぶ。
    """
    for message_id, pulse_id, piece_count, reason in notices:
        LOGGER.warning(
            "vessel voice for msg=%s closed without finish_message: %s (pieces=%d)",
            message_id, reason, piece_count,
        )
        notify_no_audio(message_id, reason, pulse_id=pulse_id)


def _vessel_held_by_addon(message_id: str) -> bool:
    """登録された ``holds`` のどれかがこの発話を持っていると答えれば真。

    例外は記録して偽として扱う。``holds`` は他のアドオンのコードなので、
    ``_vessel_lock`` を持たずに呼ぶ。
    """
    with _vessel_lock:
        holders = [holds for _, holds in _voice_claims if holds is not None]
    for holds in holders:
        try:
            if holds(message_id):
                return True
        except Exception:
            LOGGER.exception("vessel holds check failed (msg=%s)", message_id)
    return False


def _expire_vessel_messages() -> None:
    """触られないまま ``_VESSEL_MESSAGE_TTL_SEC`` を過ぎた "open" の発話を閉じる。

    閉じる前に ``_vessel_held_by_addon`` に尋ね、アドオンがまだ持っている発話は
    閉じずに触り直す。閉じた発話の吹き出しには ``notify_no_audio`` で待ち表示を
    解かせる。見回りスレッド (``_run_vessel_sweeper``) と各 API の入口から呼ばれる。
    """
    now = time.monotonic()
    with _vessel_lock:
        expired = [
            (message_id, entry)
            for message_id, entry in _vessel_messages.items()
            if entry.state == _VESSEL_OPEN
            and now - entry.touched_at > _VESSEL_MESSAGE_TTL_SEC
        ]
    if not expired:
        return
    held = {message_id for message_id, _ in expired if _vessel_held_by_addon(message_id)}
    notices: List[_ClosedNotice] = []
    with _vessel_lock:
        now = time.monotonic()
        for message_id, entry in expired:
            # 尋ねている間に締め・触り直し・別の見回りが先に済ませていたら何もしない
            if (entry.state != _VESSEL_OPEN
                    or now - entry.touched_at <= _VESSEL_MESSAGE_TTL_SEC):
                continue
            if message_id in held:
                _touch_locked(message_id, entry, now)
                continue
            notices.append((
                message_id, entry.pulse_id, len(entry.pieces),
                "the vessel never finished this message",
            ))
            _close_locked(message_id, entry, now)
    _notify_closed(notices)


def _evict_over_capacity_locked(now: float) -> List[_ClosedNotice]:
    """件数が ``_VESSEL_MESSAGE_MAX`` を超えたら、捨ててよい記録だけを捨てる。

    捨ててよいのは、終わった記録 ("unclaimed" / "closed") のうち
    ``_VESSEL_RETAIN_TERMINAL_SEC`` (1 日) より長く触られていないものだけ。
    "open" の記録は件数では捨てない (``_VESSEL_MESSAGE_TTL_SEC`` の期限切れで
    だけ閉じる)。終わった記録も 1 日は残すので、遅れて来た断片で判定がやり直され
    たり、閉じた発話が開き直されたりしない (終わった記録は切れ端を持たない
    小さな記録なので、上限を超えて残っても構わない)。
    戻り値は互換のため残している (常に空)。``_vessel_lock`` を持った状態で呼ぶ。
    """
    if len(_vessel_messages) <= _VESSEL_MESSAGE_MAX:
        return []
    for message_id, entry in list(_vessel_messages.items()):
        if len(_vessel_messages) <= _VESSEL_MESSAGE_MAX:
            break
        if (entry.state != _VESSEL_OPEN
                and now - entry.touched_at > _VESSEL_RETAIN_TERMINAL_SEC):
            del _vessel_messages[message_id]
    return []


def _run_vessel_sweeper(stop: threading.Event) -> None:
    """``_VESSEL_SWEEP_INTERVAL_SEC`` ごとに ``_expire_vessel_messages`` を回す。

    "open" の記録が無くなったら ``_vessel_sweeper`` を None に戻して抜ける。
    ``stop`` が立つか、``_vessel_sweeper`` が別の見回りに替わっていたら抜ける。
    """
    global _vessel_sweeper
    while not stop.wait(_VESSEL_SWEEP_INTERVAL_SEC):
        with _vessel_lock:
            if _vessel_sweeper is None or _vessel_sweeper.stop is not stop:
                return
        try:
            _expire_vessel_messages()
        except Exception:
            LOGGER.exception("vessel voice sweeper failed to expire messages")
        with _vessel_lock:
            if _vessel_sweeper is None or _vessel_sweeper.stop is not stop:
                return
            if not any(entry.state == _VESSEL_OPEN for entry in _vessel_messages.values()):
                _vessel_sweeper = None
                return


def _ensure_vessel_sweeper_locked() -> None:
    """見回りスレッドが動いていなければ起こす。``_vessel_lock`` を持った状態で呼ぶ。"""
    global _vessel_sweeper
    if _vessel_sweeper is not None and _vessel_sweeper.thread.is_alive():
        return
    stop = threading.Event()
    thread = threading.Thread(
        target=_run_vessel_sweeper, args=(stop,),
        name="voice-tts-vessel-sweeper", daemon=True,
    )
    _vessel_sweeper = _VesselSweeper(thread=thread, stop=stop)
    thread.start()


def _recall_claim(message_id: str, pulse_id: Optional[str]) -> Optional[bool]:
    """決まっている判定を返す (記録が無ければ None)。記録は触る。"""
    with _vessel_lock:
        entry = _vessel_messages.get(message_id)
        if entry is None:
            return None
        _touch_locked(message_id, entry, time.monotonic())
        if entry.pulse_id is None:
            entry.pulse_id = pulse_id
        return entry.state != _VESSEL_UNCLAIMED


def claim_message_voice(
    persona_id: str,
    building_id: Optional[str],
    message_id: str,
    pulse_id: Optional[str] = None,
) -> bool:
    """この発話の声を仮想身体が引き受けるかを、message_id ごとに一度だけ決める。

    ``speak_hook.on_persona_speak`` と Godot アドオンが断片ごとに呼ぶ。最初の
    呼び出しで ``_is_voice_claimed`` を評価して記録を作り (真なら "open"、偽なら
    "unclaimed")、以降は同じ答えを返す。途中で Vessel が繋がった・切れた場合も、
    記録が "closed" になった後も変えない。同じ message_id の最初の呼び出しが
    同時に来ても、判定関数は一回だけ走る (``_vessel_decision_lock``)。

    呼ぶたびに記録を触るので、断片が続いている "open" の発話は期限切れにならない。
    """
    _expire_vessel_messages()
    decided = _recall_claim(message_id, pulse_id)
    if decided is not None:
        return decided
    with _vessel_decision_lock:
        decided = _recall_claim(message_id, pulse_id)
        if decided is not None:
            # 同じ発話の別の断片が、判定の順番待ちの間に先に決めていた
            return decided
        claimed = _is_voice_claimed(persona_id, building_id or "")
        now = time.monotonic()
        with _vessel_lock:
            _vessel_messages[message_id] = _VesselMessage(
                state=_VESSEL_OPEN if claimed else _VESSEL_UNCLAIMED,
                created_at=now,
                touched_at=now,
                pulse_id=pulse_id,
            )
            evicted = _evict_over_capacity_locked(now)
            if claimed:
                _ensure_vessel_sweeper_locked()
    if claimed:
        LOGGER.debug(
            "voice of msg=%s is claimed by the vessel (persona=%s building=%s)",
            message_id, persona_id, building_id,
        )
    _notify_closed(evicted)
    return claimed


def synthesize_piece(
    persona_id: str,
    text: str,
    *,
    message_id: str,
    timeout: float,
) -> Tuple[bytes, float]:
    """仮想身体の一切れの文を合成し、``(16-bit PCM の WAV, 秒数)`` を返す。

    合成は既存のワーカースレッドに並べて行う (通常の合成とエンジンを同時に
    使わない)。声プロファイル・エンジン・話速などの合成パラメータ・読み方の
    辞書・読み上げ前の整形 (``clean_text_for_tts``) は通常の合成と同じ。
    再生・画面への通知・ファイル保存はしない。

    message_id の記録が "open" のときだけ、切れ端を依頼順に取っておき、
    ``finish_message`` で吹き出しの声にする。記録が無い・"unclaimed"・"closed"
    の発話では声を返すだけで、記録にも吹き出しにも触らない。

    Raises:
        NothingToVoice: 整形後に読み上げる文字が残らない (記録には触らない)。
            ワーカーで読み方の辞書を当てて残らなくなったときも送出し、その枠は
            ``skipped`` に入れて欠けにはしない。
        VoiceUnavailable: アドオンが無効・このペルソナの ``auto_speak`` が OFF・
            声プロファイルが無い。依頼の受付時と、ワーカーが合成を始める時の
            両方で確かめる。
        TimeoutError: ``timeout`` 秒までに合成が終わらなかった。
        その他: エンジンの例外をそのまま送出する。

    ``NothingToVoice`` 以外の例外では、"open" の発話は欠けありとして記録され、
    ``finish_message`` は吹き出しに声を付けない。
    """
    _expire_vessel_messages()
    if not clean_text_for_tts(text).strip():
        # 整形すると読み上げる文字が残らない区切り (URL だけ等) は、もともと
        # 声の無い区切り。欠けではないので、吹き出しの声を諦める理由にしない。
        raise NothingToVoice("no voiceable text in this piece")
    entry: Optional[_VesselMessage]
    slot = -1
    with _vessel_lock:
        entry = _vessel_messages.get(message_id)
        if entry is not None and entry.state == _VESSEL_OPEN:
            slot = len(entry.pieces)
            entry.pieces.append(None)
            _touch_locked(message_id, entry, time.monotonic())
        else:
            # 知らない・引き受けていない・閉じた発話。声は返すが、記録には加えない
            entry = None
    try:
        params = _get_effective_params(persona_id)
        if not params.get("_enabled", True):
            raise VoiceUnavailable("voice-tts addon is disabled")
        if not params.get("auto_speak", True):
            raise VoiceUnavailable("auto_speak is off for this persona")
        future: "Future[Tuple[np.ndarray, int]]" = Future()
        _worker.start()
        _worker._queue.put(_PieceJob(
            persona_id=persona_id, text=text, message_id=message_id, future=future,
        ))
        try:
            audio, sample_rate = future.result(timeout=timeout)
        except _FutureTimeout:
            # ワーカーがまだ始めていなければ取り消す。始めていたら合成は最後まで
            # 走るが、その結果は使わない (切れ端の枠は欠けたまま)。
            future.cancel()
            raise TimeoutError(
                f"voice piece was not synthesized within {timeout:.1f}s"
            ) from None
    except NothingToVoice:
        # ワーカーで読み方の辞書を当てたら読み上げる文字が残らなかった。入口の
        # NothingToVoice と同じく欠けにはせず、この枠をつなぐ対象から外す
        if entry is not None:
            with _vessel_lock:
                if entry.state == _VESSEL_OPEN:
                    entry.skipped.add(slot)
                    _touch_locked(message_id, entry, time.monotonic())
        raise
    except Exception:
        if entry is not None:
            with _vessel_lock:
                if entry.state == _VESSEL_OPEN:
                    entry.broken = True
                    _touch_locked(message_id, entry, time.monotonic())
        raise
    if entry is not None:
        with _vessel_lock:
            if entry.state == _VESSEL_OPEN:
                entry.pieces[slot] = (audio, sample_rate)
                _touch_locked(message_id, entry, time.monotonic())
    return _wav_bytes(audio, sample_rate), len(audio) / float(sample_rate)


def finish_message(
    message_id: str,
    *,
    complete: bool,
    pulse_id: Optional[str] = None,
) -> None:
    """仮想身体がこの発話の最後の区切りまで配送し終えたことを受け取る。

    記録が "open" のときだけ "closed" にし、吹き出しへ結末を一度だけ知らせる。
    ``complete`` が真で、切れ端が一つ以上あり、欠けた切れ端が無く、全切れ端の
    sample_rate が揃っていれば、切れ端を依頼順につないで保存し、吹き出しの
    「音声を再生」を出す。知らせには ``audio_completed`` を使う (画面の自動再生
    ``auto_play_tts`` は ``audio_ready`` だけを購読しているので、ブラウザでは
    鳴らない)。それ以外は ``notify_no_audio`` で「この吹き出しに音声は無い」を
    知らせる。

    記録が無い・"unclaimed"・"closed" の発話では何も知らせず、記録も変えない
    (二度目の呼び出しは何もしない)。"unclaimed" は通常の合成が吹き出しを持ち、
    "closed" は既に結末を知らせ済み。記録が無いのは ``claim_message_voice`` を
    経ていないか、上限で追い出された発話 (追い出しが "open" の記録だったなら、
    そのときに ``notify_no_audio`` 済み)。
    """
    _expire_vessel_messages()
    with _vessel_lock:
        entry = _vessel_messages.get(message_id)
        state = entry.state if entry is not None else None
        if entry is not None and state == _VESSEL_OPEN:
            slots = [piece for index, piece in enumerate(entry.pieces)
                     if index not in entry.skipped]
            broken = entry.broken
            stored_pulse = entry.pulse_id
            _close_locked(message_id, entry, time.monotonic())
    if entry is None or state != _VESSEL_OPEN:
        LOGGER.info(
            "finish_message: msg=%s is %s; nothing to tell the bubble",
            message_id, state or "unknown",
        )
        return
    pulse = pulse_id if pulse_id is not None else stored_pulse

    reason: Optional[str] = None
    pieces: List[Tuple[np.ndarray, int]] = []
    if not complete:
        reason = "the vessel did not deliver every piece of this message"
    elif not slots:
        reason = "no voice pieces were synthesized for this message"
    elif broken or any(piece is None for piece in slots):
        reason = "a voice piece of this message was not synthesized"
    else:
        pieces = [piece for piece in slots if piece is not None]
        if len({sample_rate for _, sample_rate in pieces}) != 1:
            reason = "voice pieces of this message have different sample rates"
    if reason is not None:
        notify_no_audio(message_id, reason, pulse_id=pulse)
        return

    sample_rate = pieces[0][1]
    full = np.concatenate([audio for audio, _ in pieces])
    version = uuid.uuid4().hex
    try:
        wav_path = _worker._save_wav(full, sample_rate, version)
    except Exception as exc:
        LOGGER.warning("Failed to save vessel voice wav: %s (msg=%s)", exc, message_id)
        notify_no_audio(message_id, f"failed to save the voice: {exc}", pulse_id=pulse)
        return
    LOGGER.debug(
        "vessel voice wav saved: %s (%d pieces, %d ms, msg=%s)",
        wav_path, len(pieces), int(len(full) / sample_rate * 1000), message_id,
    )
    _notify_audio_ready(
        message_id, wav_path,
        event_name="audio_completed",
        version=version,
        pulse_id=pulse,
    )
