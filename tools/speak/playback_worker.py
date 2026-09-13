"""Background playback worker.

Single FIFO queue + dedicated thread. synthesize -> save wav -> play via
sounddevice on the backend machine. All operations are non-blocking from the
Tool caller's perspective.
"""
from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .engine import TTSEngine, create_engine
from .profiles import get_profile
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


_OUT_DIR = _saiverse_home() / "user_data" / "voice" / "out"


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
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._engines: Dict[str, TTSEngine] = {}
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

    def _get_engine(self, name: str) -> TTSEngine:
        if name in self._engines:
            return self._engines[name]
        cfg = self._load_config()
        engine_cfg = (cfg.get("engines") or {}).get(name, {})
        engine = create_engine(name, engine_cfg)
        self._engines[name] = engine
        return engine

    def _save_wav(self, audio: np.ndarray, sample_rate: int, job_id: str) -> Path:
        import wave
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUT_DIR / f"{job_id}.wav"
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
            self._teardown_message_state(state, message_id)
            return False

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
        if hours <= 0 or not _OUT_DIR.exists():
            return
        cutoff = time.time() - hours * 3600
        for path in _OUT_DIR.glob("*.wav"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

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

        profile = get_profile(job.persona_id)
        if profile is None:
            LOGGER.info(
                "No voice profile for persona_id=%s (and no _default); skipping TTS.",
                job.persona_id,
            )
            return

        cfg = self._load_config()
        engine_name = profile.get("engine") or cfg.get("default_engine", "gpt_sovits")
        try:
            engine = self._get_engine(engine_name)
        except Exception as exc:
            LOGGER.error("Failed to initialize engine '%s': %s", engine_name, exc)
            return

        effective = _get_effective_params(job.persona_id)
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

        # ユーザー読み方辞書の適用 (TTS engine に渡す直前に文字列置換)。
        # global 辞書 + ペルソナ別オーバーライド (registry の pronunciation_dict)
        # が両方ある場合は persona の方が先に適用される。
        from . import pronunciation_dict as _pd
        persona_dict = profile.get("pronunciation_dict")
        tts_text = _pd.apply(job.text, persona_dict=persona_dict)
        if tts_text != job.text:
            LOGGER.debug(
                "pronunciation_dict applied: %d -> %d chars (persona=%s)",
                len(job.text), len(tts_text), job.persona_id,
            )

        if use_streaming:
            ok = self._play_streaming(
                engine=engine,
                text=tts_text,
                ref_audio=profile.get("ref_audio"),
                ref_text=profile.get("ref_text"),
                params=profile.get("params"),
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
                ref_audio=profile.get("ref_audio"),
                ref_text=profile.get("ref_text"),
                params=profile.get("params"),
            )
        except Exception as exc:
            LOGGER.error("TTS synthesis failed (engine=%s): %s", engine_name, exc)
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
