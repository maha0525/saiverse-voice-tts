"""addon.json manifest and persona config tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

_PACK_ROOT = Path(__file__).resolve().parent.parent
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


class AddonManifestTests(unittest.TestCase):
    def setUp(self):
        manifest_path = _PACK_ROOT / "addon.json"
        self.assertTrue(manifest_path.exists(), "addon.json must exist")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_speed_param_schema_defined(self):
        """speed parameter must be defined in params_schema for per-persona configuration."""
        params_schema = self.manifest.get("params_schema", [])
        speed_schema = next((p for p in params_schema if p.get("key") == "speed"), None)
        self.assertIsNotNone(speed_schema, "speed param must exist in params_schema")
        self.assertEqual(speed_schema.get("type"), "number")
        self.assertEqual(speed_schema.get("value_type"), "float")
        self.assertEqual(speed_schema.get("default"), 1.0)
        self.assertEqual(speed_schema.get("min"), 0.5)
        self.assertEqual(speed_schema.get("max"), 2.0)
        self.assertEqual(speed_schema.get("step"), 0.05)
        self.assertTrue(speed_schema.get("persona_configurable"))


def _addon_config_like_host(overrides):
    """Fake saiverse.addon_config whose get_params behaves like the host's.

    The host starts from addon.json's params_schema defaults and layers the
    saved global / persona values on top, so untouched fields come back with
    their default values (not missing).
    """
    manifest = json.loads((_PACK_ROOT / "addon.json").read_text(encoding="utf-8"))
    defaults = {
        p["key"]: p["default"]
        for p in manifest.get("params_schema", [])
        if "key" in p and "default" in p
    }
    module = types.ModuleType("saiverse.addon_config")
    module.get_params = lambda addon, persona_id=None: dict(
        defaults, **overrides, _enabled=True
    )
    return module


_REGISTRY = {
    "registry_persona": {
        "engine": "gpt_sovits",
        "ref_audio": "samples/registry_persona/ref.wav",
        "ref_text": "registry transcript",
        "params": {"speed": 1.3, "top_k": 20},
    },
    "registry_api_persona": {
        "engine": "openai_tts",
        "params": {"openai_voice": "nova"},
    },
    "_default": {
        "engine": "gpt_sovits",
        "ref_audio": "samples/_default/ref.wav",
        "ref_text": "default transcript",
        "params": {"speed": 1.0},
    },
}


class PersonaSpeedProfileTests(unittest.TestCase):
    def _get_profile(self, persona_id, overrides):
        from tools.speak import profiles

        with patch.dict(
            sys.modules, {"saiverse.addon_config": _addon_config_like_host(overrides)}
        ), patch.object(profiles, "_load_registry", return_value=_REGISTRY):
            return profiles.get_profile(persona_id)

    def test_speed_passes_through_to_persona_profile(self):
        """speed set together with a UI reference voice reaches the profile params."""
        profile = self._get_profile("ui_persona", {
            "ref_audio": "samples/_default/ref.wav",
            "ref_text": "UI transcript",
            "speed": 1.35,
        })
        self.assertEqual(profile["params"].get("speed"), 1.35)

    def test_ui_speed_merged_into_registry_profile(self):
        """speed set in the UI without a UI reference voice still applies to the registry voice."""
        profile = self._get_profile("registry_persona", {"speed": 1.45})
        self.assertTrue(profile["ref_audio"].endswith("ref.wav"))
        self.assertEqual(profile["params"]["speed"], 1.45)
        self.assertEqual(profile["params"]["top_k"], 20)

    def test_ui_speed_merged_into_default_profile(self):
        """speed set in the UI also applies when the persona falls back to _default."""
        profile = self._get_profile("unknown_persona", {"speed": 0.8})
        self.assertTrue(profile.get("is_default_voice"))
        self.assertEqual(profile["params"]["speed"], 0.8)

    def test_untouched_ui_fields_keep_registry_params(self):
        """UI fields left at their addon.json defaults must not overwrite registry.json params."""
        profile = self._get_profile("registry_persona", {})
        self.assertEqual(profile["params"], {"speed": 1.3, "top_k": 20})

        api_profile = self._get_profile("registry_api_persona", {})
        self.assertEqual(api_profile["params"], {"openai_voice": "nova"})
