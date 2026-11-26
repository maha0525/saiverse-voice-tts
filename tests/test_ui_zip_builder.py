"""Cross-language test for ui/zipBuilder.ts (batch parser + client ZIP builder).

Transpiles ui/zipBuilder.ts with Node + TypeScript (from SAIVerse/frontend)
or skips cleanly if Node/TypeScript is not installed, then validates:
  1. JSON string array / object array / persona_id override / error cases
  2. Line-by-line mode (commas in English sentences are preserved intact)
  3. Filename sanitization and case-insensitive deduplication
  4. Generated ZIP archive opens cleanly with Python's stdlib zipfile.ZipFile
     including UTF-8 Japanese filenames and exact CRC-32 / byte payload parity.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import unittest
import zipfile
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent.parent
_ZIP_BUILDER_TS = _PACK_ROOT / "ui" / "zipBuilder.ts"


def _find_ts_node_modules() -> Path | None:
    candidates = [
        _PACK_ROOT.parent / "SAIVerse" / "frontend" / "node_modules",
        _PACK_ROOT.parent.parent / "frontend" / "node_modules",
    ]
    for cand in candidates:
        if (cand / "typescript").exists():
            return cand
    return None


class ZipBuilderCrossLanguageTests(unittest.TestCase):
    def test_parser_and_zip_archive_valid_in_python_zipfile(self):
        node = shutil.which("node")
        node_modules = _find_ts_node_modules()
        if not node or node_modules is None:
            self.skipTest("node or typescript module not available in this environment")

        script = r"""
const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require('typescript');

const src = fs.readFileSync(process.env.ZIP_BUILDER_TS, 'utf8');
const out = ts.transpileModule(src, {
    compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
        esModuleInterop: true,
    },
}).outputText;

const mod = { exports: {} };
const fn = new Function('exports', 'require', 'module', out);
fn(mod.exports, require, mod);
const zb = mod.exports;

// 1. JSON mode: string + object + persona_id override + English commas
const r1 = zb.parseBatchInput(JSON.stringify([
    "Hello, world, how are you?",
    { filename: "挨拶", text: "おはよう！" },
    { filename: "collab", text: "いっしょに行こう", persona_id: "yui_city_a" },
]), "json");
assert.equal(r1.error, null);
assert.equal(r1.items.length, 3);
assert.equal(r1.items[0].text, "Hello, world, how are you?");
assert.equal(r1.items[1].filename, "挨拶");
assert.equal(r1.items[2].personaId, "yui_city_a");

// 2. Lines mode preserves English commas and drops blank lines
const r2 = zb.parseBatchInput("One, two, three.\n\n  Four, five!  \n", "lines");
assert.equal(r2.error, null);
assert.deepEqual(r2.items.map((x) => x.text), ["One, two, three.", "Four, five!"]);

// 3. Invalid JSON rejected cleanly
const r3 = zb.parseBatchInput("{not array}", "json");
assert.ok(r3.error);
const r4 = zb.parseBatchInput('{"a": 1}', "json");
assert.ok(r4.error);

// 4. Filename sanitization & deduplication
assert.equal(zb.sanitizeWavFilename("../a/b:c", "001"), "a_b_c.wav");
assert.equal(zb.sanitizeWavFilename("CON", "001"), "CON_.wav");
const used = new Set(["manifest.json"]);
assert.equal(zb.allocateUniqueWavName("Hi.wav", used), "Hi.wav");
assert.equal(zb.allocateUniqueWavName("hi.wav", used), "hi_2.wav");

// 4b. Single-preview WAV cache invalidation when configRevision changes or is absent
const baseKey = { personaId: "air_city_a", text: "おはよう", applyPronunciation: true, configRevision: 3 };
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey }), true);
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey, configRevision: 4 }), false);
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey, configRevision: null }), false);
assert.equal(zb.canReuseSingleSynthCache({ ...baseKey, configRevision: null }, { ...baseKey, configRevision: null }), false);
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey, text: "こんにちは" }), false);
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey, applyPronunciation: false }), false);
assert.equal(zb.canReuseSingleSynthCache(baseKey, { ...baseKey, personaId: "yui_city_a" }), false);

// 5. Build a ZIP with ASCII + Japanese UTF-8 filenames and binary WAV-like data
const wavData = new Uint8Array([0x52, 0x49, 0x46, 0x46, 0x24, 0x00, 0x00, 0x00]);
const manifestData = new TextEncoder().encode(JSON.stringify({ ok: true, label: "日本語" }));
const zipBytes = zb.buildZipBytes([
    { name: "001.wav", data: wavData },
    { name: "挨拶.wav", data: wavData },
    { name: "manifest.json", data: manifestData },
]);
process.stdout.write(Buffer.from(zipBytes).toString("base64"));
"""
        env = dict(os.environ)
        env["NODE_PATH"] = str(node_modules)
        env["ZIP_BUILDER_TS"] = str(_ZIP_BUILDER_TS)
        proc = subprocess.run(
            [node, "-e", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"node failed:\n{proc.stderr}")

        zip_bytes = base64.b64decode(proc.stdout.strip())
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            # Verify CRC-32 of every entry
            self.assertIsNone(zf.testzip())
            self.assertEqual(set(zf.namelist()), {"001.wav", "挨拶.wav", "manifest.json"})
            self.assertEqual(zf.read("001.wav"), b"RIFF$\x00\x00\x00")
            self.assertEqual(zf.read("挨拶.wav"), b"RIFF$\x00\x00\x00")
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest, {"ok": True, "label": "日本語"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
