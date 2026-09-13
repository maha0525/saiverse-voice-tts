"""Server-side hook handler for the ``persona_speak`` event.

SAIVerse 本体が ``emit_speak`` / ``emit_say`` の末尾でこの関数を呼ぶ。
``addon.json`` の ``server_hooks`` 宣言で本体に登録される。

設計は SAIVerse 本体の ``docs/intent/addon_speak_hooks.md`` を参照。

ハンドラは本体の ``ThreadPoolExecutor`` で隔離実行されるため、ここでは
``enqueue_tts()`` で内部キューに投入してすぐ return する (実合成は voice-tts
パック内の ``_TTSWorker`` スレッドで処理)。
"""
from __future__ import annotations

import logging
from typing import Any

from tools._loaded.speak.playback_worker import (
    enqueue_tts,
    get_effective_params,
    notify_no_audio,
)
from tools._loaded.speak.text_cleaner import clean_text_for_tts

LOGGER = logging.getLogger(__name__)


def on_persona_speak(
    persona_id: str,
    text_for_voice: str,
    message_id: str,
    **_kwargs: Any,
) -> None:
    """ペルソナ発話イベントを TTS キューへ投入する。

    Args:
        persona_id: 発話したペルソナの ID。
        text_for_voice: ``<in_heart>`` 除去済み・spell ブロック処理後のテキスト。
            TTS にはこれを使う。
        message_id: Building history のメッセージ ID。バブル再生ボタンや
            ``audio_ready`` SSE イベントの紐付けキー。
        **_kwargs: ``text_raw`` / ``building_id`` / ``pulse_id`` / ``source`` /
            ``metadata`` 等。本ハンドラでは未使用だが、本体側でペイロード項目が
            増えても壊れないように受ける。
    """
    if not persona_id or not message_id:
        return

    # アドオン UI の有効化トグルとペルソナ別 auto_speak フラグを尊重する。
    # アドオン全体が無効なら server_hook 自体が unregister されるため
    # ここに来ないが、念のため _enabled もチェックする。
    params = get_effective_params(persona_id)
    if not params.get("_enabled", True):
        LOGGER.debug(
            "voice-tts speak_hook skipped: addon disabled (persona=%s)",
            persona_id,
        )
        return
    if not params.get("auto_speak", True):
        LOGGER.debug(
            "voice-tts speak_hook skipped: auto_speak=false (persona=%s)",
            persona_id,
        )
        # 自動発話を切っている回も、締めの合図のときだけ「この吹き出しに音声は
        # 無い」を知らせる。知らせないと画面の音声ボタンが合図を待って回り続ける
        # (手動の再生成で後から音声が作られたら、そちらが優先して表示される)。
        if _kwargs.get("is_final", True):
            notify_no_audio(
                message_id,
                "auto_speak is off for this persona",
                pulse_id=_kwargs.get("pulse_id"),
            )
        return

    # pulse_id を audio_ready event payload まで引き渡す経路 (Phase 2 設計)。
    # subscriber 側 (frontend / stackchan) が同 pulse / 別 pulse の判定で
    # queue 末尾積み or 旧再生 preempt を切り替えるのに使う。
    pulse_id = _kwargs.get("pulse_id")

    # Pipeline Streaming (Phase 2-α): sea runtime が文区切りごとに sub-speak
    # を発火する経路で、 同 message_id に複数 sub-text が連続 enqueue される。
    # sub_seq / is_final が無ければ従来の 「1 message = 1 sub-text」 動作。
    sub_seq = _kwargs.get("sub_seq")
    is_final = _kwargs.get("is_final", True)

    # Pipeline Streaming finalize signal: ``text_for_voice=""`` + ``is_final=True``
    # は 「sub-speak で全テキスト受信済、 voice-tts 側は stream close + wav 保存
    # だけ走らせて」 という SAIVerse 本体側の依頼。 worker 側に enqueue して
    # 既存 message_state を閉じる経路に流す (合成 engine は呼ばない)。
    if not text_for_voice:
        if not is_final:
            return
        cleaned = ""
    else:
        cleaned = clean_text_for_tts(text_for_voice)
        if not cleaned and not is_final:
            return

    job_id = enqueue_tts(
        cleaned, persona_id,
        message_id=message_id, pulse_id=pulse_id,
        sub_seq=sub_seq, is_final=is_final,
    )
    LOGGER.debug(
        "voice-tts speak_hook enqueued: persona=%s msg=%s pulse=%s job=%s "
        "len=%d sub_seq=%s is_final=%s",
        persona_id, message_id, pulse_id, job_id, len(cleaned),
        sub_seq, is_final,
    )
