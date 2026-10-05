"""Irodori-TTS engine adapter (local Python inference).

Imports the upstream inference runtime from the cloned repository at
`external/Irodori-TTS/`. Model weights (Aratako/Irodori-TTS-500M-v2 +
Aratako/Semantic-DACVAE-Japanese-32dim) are downloaded automatically from
HuggingFace on first run via ``InferenceRuntime.from_key``.

Upstream code/weights license: MIT (redistribution permitted; base dependency
licenses should be individually verified before bundling).
Upstream: https://github.com/Aratako/Irodori-TTS
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

_PACK_ROOT = Path(__file__).resolve().parents[3]
_EXTERNAL_REPO = _PACK_ROOT / "external" / "Irodori-TTS"


def _prepare_sys_path() -> None:
    if not _EXTERNAL_REPO.exists():
        raise RuntimeError(
            f"Irodori-TTS repository not found at {_EXTERNAL_REPO}. "
            "Irodori-TTS was not chosen when this addon was installed: add it from "
            "the addon manager (re-open the setup options and choose Irodori-TTS), "
            "or for a manual install run: python scripts/install_backends.py irodori"
        )
    sp = str(_EXTERNAL_REPO)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# --- ストリーミング用ヘルパ (上流に触れずに疑似ストリーミングを実現する) ---

# 文末記号: これらの後で一次分割する
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?])")
# 読点: 長すぎる文を二次分割する位置
_CLAUSE_BOUNDARY = re.compile(r"(?<=[、,])")

# 合成時間見積りの定数。日本語の発話速度からの経験則。
# - budget (synthesize 時の seconds パラメータ) は余裕を持って
# - hard_trim (合成後にハードカットする長さ) はゴミが出始める境界を狙って少しタイト
# それぞれ `chars * k + margin` で算出する。
# budget は広めに取る(モデル側の truncation_factor で自然な停止を誘導するため
# budget を絞る必要がなくなった)
_BUDGET_K = 0.25
_BUDGET_MARGIN = 1.5
# hard trim は最終保険。truncation_factor で自然停止するので通常不要だが、
# 万一のゴミ末尾を刈る目的で余裕多めに残す。
_TRIM_K = 0.25
_TRIM_MARGIN = 1.5
# truncation_factor: 低確率トークンを除外して幻覚的な「予算埋めゴミ」を抑制。
# 0.75 はゴミ抑制と本文完走のバランス点(0.7 だと稀に文途中で停止、
# 0.8 以上だとゴミ再発)。実測では 45 文字までは安定完走。
_TRUNCATION_FACTOR = 0.75
# この文字数を超える文は読点でさらに分割。truncation での文途中停止を
# 回避するため、短めチャンクに割る。
_LONG_CHUNK_CHARS = 35
_INTER_CHUNK_PAUSE_SEC = 0.12  # チャンク間に挟む無音(自然な区切り)


def _split_for_streaming(text: str) -> List[str]:
    """文境界で分割し、長すぎる文は読点でさらに分割する。

    上流 Irodori は `seconds` 予算に対して実音声長が短いとゴミ末尾を埋めてくる。
    短めのチャンクに割ることで budget を実音声長に近づけられる = ゴミが減る。
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text.replace("\n", "")) if s.strip()]
    out: List[str] = []
    for s in sentences:
        if len(s) <= _LONG_CHUNK_CHARS:
            out.append(s)
            continue
        # 長い文は読点で割る。ただし割った結果がさらに短くなりすぎないよう統合する。
        clauses = [c.strip() for c in _CLAUSE_BOUNDARY.split(s) if c.strip()]
        buf = ""
        for c in clauses:
            if len(buf) + len(c) <= _LONG_CHUNK_CHARS:
                buf = (buf + c) if buf else c
            else:
                if buf:
                    out.append(buf)
                buf = c
        if buf:
            out.append(buf)
    return out


def _trim_tail_garbage(audio: np.ndarray, sample_rate: int, chars: int) -> np.ndarray:
    """Irodori が末尾に埋める破綻音声を複合的に切り詰める。

    1. 推定発話長でハードトリム (最優先)
    2. 末尾の明示的な無音 (-50dB 以下) もさらに切り詰める
    """
    if len(audio) == 0:
        return audio
    # 1. 推定長でハードトリム
    max_seconds = chars * _TRIM_K + _TRIM_MARGIN
    max_samples = int(sample_rate * max_seconds)
    if max_samples < len(audio):
        audio = audio[:max_samples]
    # 2. 末尾のほぼ無音区間を追加トリム (もともと trim_tail で消えているはずだが念のため)
    frame = max(1, sample_rate // 100)  # 10ms フレーム
    n_frames = len(audio) // frame
    if n_frames < 3:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return audio
    silence_threshold = peak * 10 ** (-50 / 20)  # -50 dB
    # 末尾からカウントダウン
    silent_tail = 0
    for i in range(n_frames - 1, -1, -1):
        seg = audio[i * frame : (i + 1) * frame]
        if float(np.max(np.abs(seg))) < silence_threshold:
            silent_tail += 1
        else:
            break
    # 150ms 以上の無音が末尾にあるときだけカット(短い余韻は残す)
    if silent_tail * 10 >= 150:
        keep_frames = n_frames - silent_tail + 5  # 50ms の余韻だけ残す
        keep_frames = max(keep_frames, 1)
        audio = audio[: keep_frames * frame]
    return audio


def _mps_available(torch_mod: Any) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


# 上流 Irodori-TTS で bf16 が使える device (inference_runtime の precision 判定)
_BF16_DEVICES = ("cuda", "xpu")


def _resolve_runtime_settings(
    config: Dict[str, Any], *, cuda_available: bool, mps_available: bool
) -> tuple[Dict[str, str], List[str]]:
    """engine 設定から、上流の RuntimeKey に渡す device と precision を決める。

    上流は使えない device (CUDA の無い機械での "cuda" 等) や、CUDA/XPU 以外での
    bf16 を ValueError で拒む。config/default.json.template は NVIDIA GPU 向け
    (cuda + bf16) なので、そのままでは Mac や GPU の無い機械で必ず失敗する。
    ここで使える device へ落とし、落とした理由を返す (呼び出し側が WARNING に出す)。

    - モデル: "cuda" が使えなければ MPS (Apple silicon)、それも無ければ CPU。
      上流の既定の device 選択 (default_runtime_device) と同じ優先順
    - コーデック: "cuda" が使えなければ CPU (上流の既定の codec_device)
    - bf16 は CUDA/XPU のときだけ。それ以外では fp32
    """
    notes: List[str] = []

    def _usable(dev: str) -> bool:
        if dev.startswith("cuda"):
            return cuda_available
        if dev == "mps":
            return mps_available
        return True

    device = str(config.get("device", "cuda"))
    if not _usable(device):
        fallback = "mps" if mps_available else "cpu"
        notes.append(f"device={device} is unavailable; using {fallback}")
        device = fallback
    model_precision = str(
        config.get("model_precision", "bf16" if device.startswith("cuda") else "fp32")
    )
    if model_precision == "bf16" and not device.startswith(_BF16_DEVICES):
        notes.append(f"bf16 is not supported on {device}; using fp32")
        model_precision = "fp32"

    codec_device = str(config.get("codec_device", "cpu"))
    if not _usable(codec_device):
        notes.append(f"codec_device={codec_device} is unavailable; using cpu")
        codec_device = "cpu"
    codec_precision = str(config.get("codec_precision", "fp32"))
    if codec_precision == "bf16" and not codec_device.startswith(_BF16_DEVICES):
        notes.append(f"codec bf16 is not supported on {codec_device}; using fp32")
        codec_precision = "fp32"

    return (
        {
            "device": device,
            "model_precision": model_precision,
            "codec_device": codec_device,
            "codec_precision": codec_precision,
        },
        notes,
    )


_SAMPLING_REQUEST_FIELDS = {
    "caption",
    "ref_latent",
    "no_ref",
    "ref_normalize_db",
    "ref_ensure_max",
    "num_candidates",
    "decode_mode",
    "seconds",
    "max_ref_seconds",
    "max_text_len",
    "max_caption_len",
    "num_steps",
    "cfg_scale_text",
    "cfg_scale_caption",
    "cfg_scale_speaker",
    "cfg_guidance_mode",
    "cfg_scale",
    "cfg_min_t",
    "cfg_max_t",
    "truncation_factor",
    "rescale_k",
    "rescale_sigma",
    "context_kv_cache",
    "speaker_kv_scale",
    "speaker_kv_min_t",
    "speaker_kv_max_layers",
    "seed",
    "trim_tail",
    "tail_window_size",
    "tail_std_threshold",
    "tail_mean_threshold",
}


class IrodoriEngine(TTSEngine):
    name = "irodori"
    # 上流 API 自体は一括合成のみだが、このアダプタは**文単位チャンキング**で
    # 疑似ストリーミングを実現する。synthesize_stream() が各文ごとに
    # 合成→後処理→yield するので、playback_worker の既存ストリーミング経路
    # (sd.OutputStream 逐次再生 + MP3 pub/sub クライアント配信) がそのまま使える。
    supports_streaming = True

    def __init__(self, engine_config: Dict[str, Any]):
        super().__init__(engine_config)
        self._runtime = None

    def _lazy_load(self) -> None:
        if self._runtime is not None:
            return
        _prepare_sys_path()
        try:
            from irodori_tts.inference_runtime import (  # type: ignore
                InferenceRuntime,
                RuntimeKey,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import irodori_tts.inference_runtime. Ensure external/Irodori-TTS "
                "is cloned and its requirements are installed."
            ) from exc

        checkpoint = self.config.get("checkpoint", "Aratako/Irodori-TTS-500M-v2")
        codec_repo = self.config.get("codec_repo", "Aratako/Semantic-DACVAE-Japanese-32dim")
        import torch  # type: ignore  # 上の import で読み込み済み
        settings, notes = _resolve_runtime_settings(
            self.config,
            cuda_available=torch.cuda.is_available(),
            mps_available=_mps_available(torch),
        )
        for note in notes:
            LOGGER.warning("Irodori-TTS: %s", note)
        device = settings["device"]
        model_precision = settings["model_precision"]
        codec_device = settings["codec_device"]
        codec_precision = settings["codec_precision"]

        # 上流 InferenceRuntime は checkpoint をローカルファイルパスとして解釈する
        # ため、HF repo ID が指定された場合はここで model.safetensors を解決する。
        checkpoint_path = self._resolve_checkpoint(checkpoint)

        LOGGER.info(
            "Loading Irodori-TTS runtime: %s (device=%s precision=%s)",
            checkpoint, device, model_precision,
        )
        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint_path,
                model_device=device,
                model_precision=model_precision,
                codec_device=codec_device,
                codec_precision=codec_precision,
                codec_repo=codec_repo,
            )
        )

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        suffix = Path(checkpoint).suffix.lower()
        if suffix in {".pt", ".safetensors"}:
            return checkpoint
        # HuggingFace repo ID と見なしてダウンロード
        from huggingface_hub import hf_hub_download  # type: ignore
        filename = "model.safetensors"
        resolved = hf_hub_download(repo_id=checkpoint, filename=filename)
        LOGGER.info("Irodori-TTS checkpoint resolved: hf://%s -> %s", checkpoint, resolved)
        return resolved

    def synthesize(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        self._lazy_load()
        params = params or {}

        if not ref_audio:
            raise ValueError("Irodori-TTS zero-shot cloning requires ref_audio.")

        from irodori_tts.inference_runtime import SamplingRequest  # type: ignore

        # registry.json の params から SamplingRequest のフィールドだけ抜き出す。
        # ref_text は上流がサポートしていないため黙って無視(ref_wav のみで音声特徴を推定)。
        kwargs = {k: v for k, v in params.items() if k in _SAMPLING_REQUEST_FIELDS}

        result = self._runtime.synthesize(
            SamplingRequest(
                text=text,
                ref_wav=ref_audio,
                **kwargs,
            )
        )

        # SamplingResult.audio は torch.Tensor (shape: [samples] or [channels, samples])。
        # bf16 運用時は numpy が BFloat16 を扱えないので、明示的に float32 にキャスト。
        import torch  # local import: lazy_load 経由で既に存在する
        audio_tensor = result.audio
        if hasattr(audio_tensor, "detach"):
            audio_tensor = audio_tensor.detach().to(torch.float32).cpu()
        audio = np.asarray(audio_tensor, dtype=np.float32)
        if audio.ndim > 1:
            # ステレオは平均して mono 化 (playback_worker は mono 前提)
            audio = audio.mean(axis=0) if audio.shape[0] < audio.shape[-1] else audio[:, 0]
        sr = int(result.sample_rate)
        duration_ms = int(len(audio) / sr * 1000)
        return SynthesisResult(audio=audio, sample_rate=sr, duration_ms=duration_ms)

    def synthesize_stream(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SynthesisChunk]:
        """文単位チャンキングによる疑似ストリーミング。

        各チャンクで合成完了を待たずに yield するので、playback_worker 側は
        最初のチャンク分 (約 1 秒) で再生開始できる。後続チャンクは GPU が
        合成速度 RTF ~0.3 で動くため、再生が追い越すことはない。
        """
        if not text.strip():
            return
        chunks = _split_for_streaming(text)
        if not chunks:
            return

        base_params = dict(params or {})
        sample_rate_cache: Optional[int] = None

        for i, chunk_text in enumerate(chunks):
            # チャンクの文字数に応じて seconds 予算を設定する。上流の既定 30 秒で
            # 丸投げすると予算を埋めるためにゴミ末尾が大量発生するため、ここで絞る。
            # ただしタイトすぎると単語の途中で切れるので余裕を持たせる。
            n_chars = len(chunk_text)
            budget = max(3.0, min(30.0, n_chars * _BUDGET_K + _BUDGET_MARGIN))

            chunk_params = dict(base_params)
            chunk_params["seconds"] = budget
            # truncation_factor はゴミ末尾対策で最重要(モデルが低確率トークン
            # で幻覚を出すのを抑制し、自然な箇所で停止する)。params で上書き
            # されていなければ既定値を適用。
            chunk_params.setdefault("truncation_factor", _TRUNCATION_FACTOR)
            # 上流の trim_tail を強めに効かせる (SamplingRequest 既定より厳しく)
            chunk_params.setdefault("trim_tail", True)
            chunk_params.setdefault("tail_std_threshold", 0.08)
            chunk_params.setdefault("tail_mean_threshold", 0.15)

            try:
                result = self.synthesize(
                    text=chunk_text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    params=chunk_params,
                )
            except Exception as exc:
                # 1 チャンク失敗しても他は続行する (部分再生 > 全消失)
                LOGGER.warning(
                    "Irodori chunk %d/%d failed (chars=%d): %s",
                    i + 1, len(chunks), n_chars, exc,
                )
                continue

            audio = _trim_tail_garbage(result.audio, result.sample_rate, n_chars)
            if sample_rate_cache is None:
                sample_rate_cache = result.sample_rate

            if len(audio) > 0:
                yield SynthesisChunk(audio=audio, sample_rate=result.sample_rate)

            # 文間に短い無音を挟む(句点後の自然な息継ぎになる + 後続チャンクの
            # 冒頭ゴミと前チャンクの末尾ゴミが連続して聞こえづらくなる副次効果)。
            # 最終チャンクの後には付けない。
            if i < len(chunks) - 1 and sample_rate_cache is not None:
                pause_samples = int(sample_rate_cache * _INTER_CHUNK_PAUSE_SEC)
                if pause_samples > 0:
                    yield SynthesisChunk(
                        audio=np.zeros(pause_samples, dtype=np.float32),
                        sample_rate=sample_rate_cache,
                    )
