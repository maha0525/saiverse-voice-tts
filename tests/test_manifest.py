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

    def test_speed_passes_through_to_persona_profile(self):
        """speed configured in addon_config must be passed through to profile params."""
        from tools.speak import profiles

        addon_config = types.ModuleType("saiverse.addon_config")
        addon_config.get_params = lambda addon, persona_id=None: {
            "engine": "gpt_sovits",
            "ref_audio": "samples/_default/ref.wav",
            "ref_text": "テスト書き起こし",
            "speed": 1.35,
        }

        with patch.dict(sys.modules, {"saiverse.addon_config": addon_config}):
            profile = profiles.get_profile("test_persona")
            self.assertIsNotNone(profile)
            self.assertEqual(profile["params"].get("speed"), 1.35)

    def test_speed_merged_into_registry_fallback_profile(self):
        """speed configured in UI must be merged into registry-backed profile when ref_audio is omitted in UI."""
        from tools.speak import profiles

        # Persona has speed override in UI, but no ref_audio (uses registry fallback)
        addon_config = types.ModuleType("saiverse.addon_config")
        addon_config.get_params = lambda addon, persona_id=None: {
            "speed": 1.45,
        }

        with patch.dict(sys.modules, {"saiverse.addon_config": addon_config}):
            profile = profiles.get_profile("fallback_persona")
            self.assertIsNotNone(profile, "Profile should fall back to registry/_default")
            # Must retain registry audio while adopting the UI-configured speed
            self.assertIsNotNone(profile.get("ref_audio"))
            self.assertEqual(profile["params"].get("speed"), 1.45)
