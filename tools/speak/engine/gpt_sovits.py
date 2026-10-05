"""GPT-SoVITS engine adapter (local Python inference — no external API server).

Imports the upstream `TTS_infer_pack.TTS` module from the cloned repository at
`external/GPT-SoVITS/`. Pretrained weights are downloaded from HuggingFace
`lj1995/GPT-SoVITS` and placed under `external/GPT-SoVITS/GPT_SoVITS/pretrained_models/`.

Upstream code/weights license: MIT (redistribution permitted).
Upstream: https://github.com/RVC-Boss/GPT-SoVITS
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from .base import SynthesisChunk, SynthesisResult, TTSEngine

LOGGER = logging.getLogger(__name__)

_PACK_ROOT = Path(__file__).resolve().parents[3]
_EXTERNAL_REPO = _PACK_ROOT / "external" / "GPT-SoVITS"


def _prepare_sys_path() -> None:
    if not _EXTERNAL_REPO.exists():
        raise RuntimeError(
            f"GPT-SoVITS repository not found at {_EXTERNAL_REPO}. "
            "GPT-SoVITS was not chosen when this addon was installed: add it from "
            "the addon manager (re-open the setup options and choose GPT-SoVITS), "
            "or for a manual install run: python scripts/install_backends.py gpt_sovits"
        )
    for p in (_EXTERNAL_REPO, _EXTERNAL_REPO / "GPT_SoVITS"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _mps_available(torch_mod: Any) -> bool:
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    try:
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def _resolve_device(
    desired: str, *, cuda_available: bool, mps_available: bool
) -> tuple[str, bool]:
    """engine 設定の device から、実際に使う (device, is_half) を決める。

    - "cuda": 使えれば半精度 (従来どおり)。使えなければ CPU
    - "mps" (Apple silicon の GPU): 使えれば全精度。上流 GPT-SoVITS の半精度の
      扱いは「CPU なら切る」だけで MPS を考慮していない (TTS_Config)。動作未確認の
      Mac で数値が崩れる要因を増やさないよう、MPS では is_half を必ず切る。
      使えなければ CPU
    - それ以外 ("cpu" を含む): CPU、全精度

    "cuda" が使えないときに MPS へ自動で切り替えはしない。上流の推論 WebUI も
    MPS の自動選択をコメントアウトしていて (inference_webui_fast.py)、MPS は
    設定で明示したときだけ使う。
    """
    if desired == "cuda" and cuda_available:
        return "cuda", True
    if desired == "mps" and mps_available:
        return "mps", False
    return "cpu", False


@contextlib.contextmanager
def _cwd(path: Path) -> Iterator[None]:
    """Temporarily chdir to ``path``. GPT-SoVITS uses relative paths internally
    (pretrained_models/..., configs/...), so we must be inside the repo root
    during both initialization and inference."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


class GPTSoVITSEngine(TTSEngine):
    name = "gpt_sovits"
    supports_streaming = True

    def __init__(self, engine_config: Dict[str, Any]):
        super().__init__(engine_config)
        self._tts = None
        self._last_ref: Optional[str] = None
        self._ref_language = self.config.get("ref_language", "ja")
        self._target_language = self.config.get("target_language", "ja")

    _torchaudio_patched = False

    @classmethod
    def _patch_torchaudio_load(cls) -> None:
        """Monkey-patch ``torchaudio.load`` to fall back to soundfile.

        torchaudio 2.11+ (shipped with torch 2.11 / Python 3.13) deprecated
        ``set_audio_backend`` and forces torchcodec as the default decoder.
        torchcodec requires FFmpeg shared libraries (DLLs) which are rarely
        present on Windows. Rather than requiring users to install FFmpeg,
        we wrap ``torchaudio.load`` so that any failure (torchcodec missing,
        DLL not found, etc.) transparently falls back to ``soundfile.read``.
        """
        if cls._torchaudio_patched:
            return
        cls._torchaudio_patched = True
        try:
            import torchaudio  # type: ignore
            import soundfile as sf  # type: ignore
            import torch  # type: ignore

            _original = torchaudio.load

            def _patched_load(filepath, *args, **kwargs):
                try:
                    return _original(filepath, *args, **kwargs)
                except Exception:
                    data, sr = sf.read(str(filepath), dtype="float32")
                    if data.ndim == 1:
                        tensor = torch.from_numpy(data).unsqueeze(0)
                    else:
                        tensor = torch.from_numpy(data.T)
                    return tensor, sr

            torchaudio.load = _patched_load
            LOGGER.debug("torchaudio.load patched with soundfile fallback")
        except ImportError:
            pass

    def _lazy_load(self) -> None:
        if self._tts is not None:
            return
        _prepare_sys_path()

        if self.config.get("device") == "mps":
            # MPS に未実装の演算を CPU で代行させる。torch の import より前に
            # 立てる必要がある (子プロセスではここより前に torch を import しない)。
            # 上流の inference_webui_fast.py にも同じ指定がコメントで残っている。
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # torchaudio 2.11+ (torch 2.11, Python 3.13) では set_audio_backend が
        # deprecated no-op になり、torchcodec が強制されるが Windows では
        # FFmpeg DLL 依存で動作しない。torchaudio.load を monkey-patch して
        # soundfile にフォールバックすることで、torchcodec も FFmpeg も不要にする。
        self._patch_torchaudio_load()

        # GPT-SoVITS reads pretrained_models/... via relative paths, so init
        # must happen with cwd == repo root. The host's `tools` package
        # collision (GPT-SoVITS internal `from tools.audio_sr import ...` vs
        # SAIVerse's tools/) is handled by the host's addon_external_loader,
        # which redirects external/ callers to addons.<addon>.tools transparently.
        with _cwd(_EXTERNAL_REPO):
            try:
                from TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Failed to import GPT-SoVITS TTS_infer_pack. "
                    "Ensure external/GPT-SoVITS is properly installed and its "
                    "dependencies are available."
                ) from exc

            config_yaml = self.config.get("config_yaml")
            if config_yaml:
                cfg = TTS_Config(str(config_yaml))
            else:
                cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")

            # tts_infer.yaml の全セクションで device: cpu がハードコードされている
            # ため、CUDA が利用可能でも CPU 推論になってしまう。
            # 上流ファイルを変更せず、ここで cfg を上書きして CUDA を有効化する。
            # engine 設定の device ("cuda" 既定 / "mps" / "cpu") で明示的に選べる
            # (Irodori-TTS の device と同じ扱い)。
            import torch  # type: ignore
            desired = self.config.get("device", "cuda")
            device, is_half = _resolve_device(
                desired,
                cuda_available=torch.cuda.is_available(),
                mps_available=_mps_available(torch),
            )
            cfg.device = device
            cfg.is_half = is_half
            if device == desired:
                LOGGER.info(
                    "Loading GPT-SoVITS TTS pipeline (%s, %s precision)",
                    device, "half" if is_half else "full",
                )
            else:
                LOGGER.warning(
                    "Loading GPT-SoVITS TTS pipeline (CPU, full precision) "
                    "— %s requested but unavailable; inference will be very "
                    "slow. Install a torch build for that device for GPU "
                    "acceleration.",
                    desired,
                )

            self._tts = TTS(cfg)

    def _build_inputs(
        self,
        text: str,
        ref_audio: str,
        ref_text: Optional[str],
        params: Dict[str, Any],
        *,
        streaming: bool,
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {
            "text": text,
            "text_lang": self._target_language,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_text or "",
            "prompt_lang": self._ref_language,
            "text_split_method": params.get("text_split_method", "cut5"),
            "batch_size": 1,
            "speed_factor": float(params.get("speed", 1.0)),
            "top_k": int(params.get("top_k", 15)),
            "top_p": float(params.get("top_p", 1.0)),
            "temperature": float(params.get("temperature", 1.0)),
        }
        if streaming:
            # GPT-SoVITS streaming mode requires parallel_infer=False and
            # is unavailable for V3/V4 vocoder models (auto-falls back to
            # return_fragment). See TTS.run() docstring for details.
            inputs.update(
                {
                    "streaming_mode": True,
                    "parallel_infer": False,
                    "return_fragment": False,
                    "overlap_length": int(params.get("overlap_length", 2)),
                    "min_chunk_length": int(params.get("min_chunk_length", 16)),
                    "fixed_length_chunk": bool(params.get("fixed_length_chunk", False)),
                }
            )
        else:
            inputs["return_fragment"] = False
        return inputs

    @staticmethod
    def _normalize_chunk(audio_chunk: Any) -> np.ndarray:
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.asarray(audio_chunk)
        if audio_chunk.dtype == np.int16:
            return audio_chunk.astype(np.float32) / 32768.0
        return audio_chunk.astype(np.float32)

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
            raise ValueError("GPT-SoVITS requires ref_audio.")

        inputs = self._build_inputs(text, ref_audio, ref_text, params, streaming=False)

        with _cwd(_EXTERNAL_REPO):
            if ref_audio != self._last_ref:
                self._tts.set_ref_audio(ref_audio)
                self._last_ref = ref_audio

            chunks: list[np.ndarray] = []
            sr = 32000
            for sr_chunk, audio_chunk in self._tts.run(inputs):
                sr = int(sr_chunk)
                chunks.append(self._normalize_chunk(audio_chunk))

        if not chunks:
            raise RuntimeError("GPT-SoVITS produced no audio.")

        audio = np.concatenate(chunks)
        duration_ms = int(len(audio) / sr * 1000)
        return SynthesisResult(audio=audio, sample_rate=sr, duration_ms=duration_ms)

    def synthesize_stream(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SynthesisChunk]:
        self._lazy_load()
        params = params or {}

        if not ref_audio:
            raise ValueError("GPT-SoVITS requires ref_audio.")

        inputs = self._build_inputs(text, ref_audio, ref_text, params, streaming=True)

        with _cwd(_EXTERNAL_REPO):
            if ref_audio != self._last_ref:
                self._tts.set_ref_audio(ref_audio)
                self._last_ref = ref_audio

            for sr_chunk, audio_chunk in self._tts.run(inputs):
                yield SynthesisChunk(
                    audio=self._normalize_chunk(audio_chunk),
                    sample_rate=int(sr_chunk),
                )
