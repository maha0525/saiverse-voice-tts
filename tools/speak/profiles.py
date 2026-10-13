"""Voice profile registry loader.

Maps persona_id to voice profile entry with 3-tier fallback:

    1. AddonPersonaConfig (host UI で設定した参照音声) — 最優先
    2. voice_profiles/registry.json の該当ペルソナエントリ
    3. voice_profiles/registry.json の "_default" エントリ

いずれにも無い場合は None を返し、TTS はスキップされる。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

_PACK_ROOT = Path(__file__).resolve().parent.parent.parent
# ユーザーローカル版(.gitignore 対象)。存在しなければ .template から自動コピー。
_REGISTRY_PATH = _PACK_ROOT / "voice_profiles" / "registry.json"
_REGISTRY_TEMPLATE_PATH = _PACK_ROOT / "voice_profiles" / "registry.json.template"
_ADDON_NAME = "saiverse-voice-tts"

_cached_registry: Optional[Dict[str, Any]] = None


def _load_registry() -> Dict[str, Any]:
    global _cached_registry
    if _cached_registry is not None:
        return _cached_registry
    # ローカル版が無ければ .template から初回コピー (first-run materialization)。
    if not _REGISTRY_PATH.exists() and _REGISTRY_TEMPLATE_PATH.exists():
        try:
            _REGISTRY_PATH.write_text(
                _REGISTRY_TEMPLATE_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            LOGGER.info(
                "Materialized %s from %s (first run).",
                _REGISTRY_PATH.name, _REGISTRY_TEMPLATE_PATH.name,
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed to materialize %s from template: %s",
                _REGISTRY_PATH, exc,
            )
    # 読み込み元: ローカル版 → なければ template
    source = _REGISTRY_PATH if _REGISTRY_PATH.exists() else _REGISTRY_TEMPLATE_PATH
    if not source.exists():
        LOGGER.warning(
            "Voice profile registry not found (neither %s nor %s)",
            _REGISTRY_PATH, _REGISTRY_TEMPLATE_PATH,
        )
        _cached_registry = {}
        return _cached_registry
    try:
        _cached_registry = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.error("Failed to parse voice profile registry from %s: %s", source, exc)
        _cached_registry = {}
    return _cached_registry


def _resolve_ref_audio(ref_audio: Optional[str]) -> Optional[str]:
    """ref_audio パスを絶対パスに解決する。

    - 絶対パスはそのまま返す
    - voice_profiles/ 相対パスは pack root から解決
    - 存在しないパスはそのまま返す（エンジン側でエラーになる）
    """
    if not ref_audio:
        return None
    p = Path(ref_audio)
    if p.is_absolute():
        return str(p)
    resolved = (_PACK_ROOT / "voice_profiles" / ref_audio).resolve()
    return str(resolved)


_API_ENGINES = {"openai_tts", "elevenlabs"}


# 以下は合成パラメータ (profile の params) に含めず除外:
# - addon フレームワーク内部 / UI トグル ( _enabled, server_side_playback 等)
# - profile top-level に独立して載るもの (ref_audio, ref_text, engine)
# - pronunciation_dict (addon UI ではグローバル設定。pronunciation_dict.py
#   が直接ロードする)
# - API key 系・外部サーバー接続設定 (engine が addon_config から fresh に解決する)
_EXCLUDED_KEYS = {
    "_enabled", "ref_audio", "ref_text", "engine",
    "auto_speak", "server_side_playback", "client_side_playback",
    "streaming", "output_device", "pronunciation_dict",
    "openai_api_key", "elevenlabs_api_key",
    "gpt_sovits_remote_url", "gpt_sovits_remote_url_override",
    "gpt_sovits_remote_timeout_sec",
}

_cached_manifest_defaults: Optional[Dict[str, Any]] = None


def _extract_synthesis_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not params:
        return {}
    return {k: v for k, v in params.items() if k not in _EXCLUDED_KEYS and v is not None}


def _load_manifest_defaults() -> Dict[str, Any]:
    """addon.json の params_schema の既定値 (key -> default)。"""
    global _cached_manifest_defaults
    if _cached_manifest_defaults is not None:
        return _cached_manifest_defaults
    defaults: Dict[str, Any] = {}
    try:
        manifest = json.loads((_PACK_ROOT / "addon.json").read_text(encoding="utf-8"))
        for param in manifest.get("params_schema", []):
            key = param.get("key")
            if key and "default" in param:
                defaults[key] = param["default"]
    except Exception as exc:
        LOGGER.warning("Failed to read params_schema defaults from addon.json: %s", exc)
    _cached_manifest_defaults = defaults
    return defaults


def _changed_synthesis_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """UI で既定値から変えられた合成パラメータだけを返す。

    host の get_params は、誰も触っていない欄にも addon.json の既定値を
    入れて返す。既定値のままの欄まで registry.json のエントリに重ねると、
    registry.json に書いた話速や声が黙って既定値に戻るため、既定値と
    異なる値だけを重ねる対象にする。
    """
    defaults = _load_manifest_defaults()
    return {
        k: v
        for k, v in _extract_synthesis_params(params).items()
        if k not in defaults or v != defaults[k]
    }


def _load_addon_persona_params(persona_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not persona_id:
        return None
    try:
        from saiverse.addon_config import get_params  # type: ignore

        return get_params(_ADDON_NAME, persona_id=persona_id)
    except Exception as exc:
        LOGGER.debug("addon_config unavailable for persona profile: %s", exc)
        return None


def _build_addon_persona_profile(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not params:
        return None
    engine_raw = params.get("engine") or "gpt_sovits"
    engine = str(engine_raw).strip() or "gpt_sovits"
    ref_audio = params.get("ref_audio")
    remote_gpt_sovits = engine == "gpt_sovits" and bool(
        str(params.get("gpt_sovits_remote_url_override") or "").strip()
        or str(params.get("gpt_sovits_remote_url") or "").strip()
    )
    if engine not in _API_ENGINES and not remote_gpt_sovits and not ref_audio:
        return None
    ref_text = params.get("ref_text", "")
    return {
        "engine": engine,
        "_server_profile_only": bool(remote_gpt_sovits and not ref_audio),
        "ref_audio": str(ref_audio) if ref_audio else None,
        "ref_text": ref_text,
        "params": _extract_synthesis_params(params),
    }


def _try_addon_persona_config(persona_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """AddonPersonaConfig から参照音声 / API エンジン設定を取得する (最優先)。

    プロファイルとして有効と判定する条件:
    - ローカルエンジン (gpt_sovits / irodori): ``ref_audio`` がセットされている
    - 外部 GPT-SoVITS: URL がある (サーバーに登録済みの参照音声も利用可能)
    - API エンジン (openai_tts / elevenlabs): ``ref_audio`` 不要
      (ref_audio は無視される。代わりに voice / voice_id がエンジン側で必要)

    上記いずれにも該当しなければ None を返し、後続の registry.json
    フォールバックに移る。
    """
    params = _load_addon_persona_params(persona_id)
    return _build_addon_persona_profile(params)


def get_profile(persona_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the voice profile for persona_id.

    Fallback order:
        1. AddonPersonaConfig (UI-managed ref_audio/ref_text)
        2. registry.json[persona_id]
        2b. AddonPersonaConfig with a remote GPT-SoVITS URL but no ref_audio
            (the server's already-registered profile for this persona)
        3. registry.json["_default"]
        4. None (TTS skipped)

    2b sits before ``_default`` so the sample voice is never uploaded to a
    shared server under this persona's ID (the remote engine updates server
    profiles to match the local reference).
    """
    raw_params = _load_addon_persona_params(persona_id)
    ui_synthesis_params = _changed_synthesis_params(raw_params)

    # Tier 1: AddonPersonaConfig
    addon_profile = _build_addon_persona_profile(raw_params)
    server_only: Optional[Dict[str, Any]] = None
    if addon_profile is not None:
        if addon_profile.pop("_server_profile_only", False):
            server_only = addon_profile
        else:
            addon_profile["ref_audio"] = _resolve_ref_audio(addon_profile.get("ref_audio"))
            LOGGER.debug("profile source=addon_config persona=%s", persona_id)
            return addon_profile

    # Tier 2 / 3: registry.json
    registry = _load_registry()
    entry = None
    if persona_id and persona_id in registry:
        entry = dict(registry[persona_id])
        LOGGER.debug("profile source=registry persona=%s", persona_id)
    elif server_only is not None:
        LOGGER.debug("profile source=remote_server_profile persona=%s", persona_id)
        return server_only
    elif "_default" in registry:
        entry = dict(registry["_default"])
        # 見本の声。ペルソナ ID でサーバーに声を登録するエンジンには渡さない
        # (playback_worker が参照音声を外す)。
        entry["is_default_voice"] = True
        LOGGER.debug("profile source=registry_default persona=%s", persona_id)

    if entry is None:
        return None

    # UI で既定値から変えられた合成パラメータ (ペルソナ別の話速など) だけを
    # registry.json のエントリに重ねる。既定値のままの欄は registry.json を尊重する。
    if ui_synthesis_params:
        merged_params = dict(entry.get("params") or {})
        merged_params.update(ui_synthesis_params)
        entry["params"] = merged_params

    entry["ref_audio"] = _resolve_ref_audio(entry.get("ref_audio"))
    return entry


def reload_registry() -> None:
    global _cached_registry
    _cached_registry = None
