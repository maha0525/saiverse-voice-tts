"""TTS engine registry."""
from __future__ import annotations

import os
from typing import Any, Dict

from .base import SynthesisResult, TTSEngine


def create_engine(name: str, engine_config: Dict[str, Any]) -> TTSEngine:
    name = (name or "").lower()
    if name == "gpt_sovits":
        # gpt_sovits は重いローカル推論で本体プロセスの GIL を奪われると
        # 激遅になるため、常に別プロセスで動かす (代理エンジン経由)。
        # docs/out_of_process.md
        # 子プロセス側は GPTSoVITSEngine を直接使う (再帰回避)。
        if os.environ.get("VOICE_TTS_IN_PROCESS") == "1":
            from .gpt_sovits import GPTSoVITSEngine
            return GPTSoVITSEngine(engine_config)
        from .subprocess_proxy import SubprocessGPTSoVITSEngine
        return SubprocessGPTSoVITSEngine(engine_config)
    if name == "irodori":
        from .irodori import IrodoriEngine
        return IrodoriEngine(engine_config)
    if name == "openai_tts":
        from .openai_tts import OpenAITTSEngine
        return OpenAITTSEngine(engine_config)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsEngine
        return ElevenLabsEngine(engine_config)
    raise ValueError(f"Unknown TTS engine: {name}")


__all__ = ["TTSEngine", "SynthesisResult", "create_engine"]
