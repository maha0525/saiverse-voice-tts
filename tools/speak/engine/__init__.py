"""TTS engine registry."""
from __future__ import annotations

import os
from typing import Any, Dict

from .base import SynthesisResult, TTSEngine


def create_engine(name: str, engine_config: Dict[str, Any]) -> TTSEngine:
    name = (name or "").lower()
    if name in ("gpt_sovits", "irodori"):
        # ローカル合成 (gpt_sovits / irodori) は常に別プロセスで動かす (代理
        # エンジン経由)。理由は二つ:
        # - 重いローカル推論で本体プロセスの GIL を奪われると激遅になる
        #   (docs/out_of_process.md)
        # - アドオンカタログで導入した場合、エンジンのパッケージはエンジン専用の
        #   Python 環境にだけ入っていて、本体プロセスでは import できない
        # 子プロセス側 (subprocess_worker) はエンジンクラスを直接使う (再帰回避)。
        # VOICE_TTS_IN_PROCESS=1 は旧来の手動導入 (本体 venv に全部入り) 向けの
        # 逃げ道で、本体プロセスの中で直接合成する。
        if os.environ.get("VOICE_TTS_IN_PROCESS") == "1":
            if name == "gpt_sovits":
                from .gpt_sovits import GPTSoVITSEngine
                return GPTSoVITSEngine(engine_config)
            from .irodori import IrodoriEngine
            return IrodoriEngine(engine_config)
        from .subprocess_proxy import SubprocessTTSEngine
        return SubprocessTTSEngine(name, engine_config)
    if name == "openai_tts":
        from .openai_tts import OpenAITTSEngine
        return OpenAITTSEngine(engine_config)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsEngine
        return ElevenLabsEngine(engine_config)
    raise ValueError(f"Unknown TTS engine: {name}")


__all__ = ["TTSEngine", "SynthesisResult", "create_engine"]
