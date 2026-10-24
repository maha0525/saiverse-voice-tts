"""requirements.txt と envs/*.txt の下限のテスト。

- PyPI から入るパッケージの行は、すべて下限 (>= か ==) を持つ
- 脆弱性の修正版を下限にしたパッケージ (certifi / aiohttp / pillow / protobuf / nltk / brotli) が
  下限を下げられていない
- 同じエンジンの cuda 用と macOS 用のファイルで、同じパッケージの下限が揃っている
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent.parent
_ENVS = _PACK_ROOT / "envs"
_REQUIREMENT_FILES = [_PACK_ROOT / "requirements.txt", *sorted(_ENVS.glob("*.txt"))]

_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pypi_requirements(path: Path) -> dict[str, list[str]]:
    """PyPI から入る行の {正規化した名前: 版の指定の並び}。

    コメント・pip のオプション (--extra-index-url 等)・ローカルのパス (./...)・
    入手先を URL で与える行 (name @ git+...) は対象外。marker (; 以降) は外す。
    同じ名前が marker 違いで複数行あれば、版の指定を全部集める。
    """
    out: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or line.startswith("."):
            continue
        if " @ " in line:
            continue
        line = line.split(";", 1)[0].strip()
        m = _NAME_RE.match(line)
        assert m is not None, f"{path.name}: 読めない行: {raw!r}"
        specs = [s.strip() for s in m.group(3).split(",") if s.strip()]
        out.setdefault(_normalize(m.group(1)), []).extend(specs)
    return out


def _version_tuple(text: str) -> tuple:
    base = text.split("+", 1)[0]
    return tuple(int(p) for p in re.findall(r"\d+", base))


def _lower_bound(specs: list[str]) -> str | None:
    for s in specs:
        if s.startswith(">=") or s.startswith("=="):
            return s[2:].strip()
    return None


# 脆弱性の修正版を下限にしたパッケージ: {ファイル名: {パッケージ: 下限の最小値}}。
# 下限を上げるのは自由だが、ここより下げると修正前の版が入り得る。
_SECURITY_FLOORS = {
    "requirements.txt": {"requests": "2.33.0"},
    "gpt_sovits-cuda.txt": {
        "certifi": "2026.10.19",
        "pillow": "13.0.0",
        "aiohttp": "3.14.4",
        "protobuf": "6.33.5",
        "nltk": "3.10.4",
    },
    "gpt_sovits-macos.txt": {
        "certifi": "2026.10.19",
        "pillow": "13.0.0",
        "aiohttp": "3.14.4",
        "protobuf": "6.33.5",
        "nltk": "3.10.4",
    },
    "irodori-cuda.txt": {
        "certifi": "2026.10.19",
        "pillow": "12.3.0",
        "aiohttp": "3.14.4",
        "protobuf": "4.25.8",
        "brotli": "1.2.1",
    },
    "irodori-macos.txt": {
        "certifi": "2026.10.19",
        "pillow": "12.3.0",
        "aiohttp": "3.14.4",
        "protobuf": "4.25.8",
        "brotli": "1.2.1",
    },
}


class DependencyFloorTests(unittest.TestCase):
    def test_every_pypi_requirement_has_a_lower_bound(self):
        for path in _REQUIREMENT_FILES:
            for name, specs in _pypi_requirements(path).items():
                with self.subTest(file=path.name, package=name):
                    self.assertIsNotNone(
                        _lower_bound(specs),
                        f"{path.name} の {name} に下限 (>= か ==) が無い: {specs}",
                    )

    def test_security_floors_are_not_lowered(self):
        for file_name, floors in _SECURITY_FLOORS.items():
            path = _PACK_ROOT / file_name if file_name == "requirements.txt" else _ENVS / file_name
            reqs = _pypi_requirements(path)
            for package, minimum in floors.items():
                with self.subTest(file=file_name, package=package):
                    self.assertIn(package, reqs, f"{file_name} に {package} の行が無い")
                    bound = _lower_bound(reqs[package])
                    self.assertIsNotNone(bound)
                    self.assertGreaterEqual(_version_tuple(bound), _version_tuple(minimum))

    def test_cuda_and_macos_files_share_floors(self):
        # torch / torchaudio (CUDA 版と PyPI 版) と onnxruntime(-gpu) は OS で行が違う
        os_specific = {"torch", "torchaudio", "onnxruntime", "onnxruntime-gpu"}
        for engine in ("gpt_sovits", "irodori"):
            cuda = _pypi_requirements(_ENVS / f"{engine}-cuda.txt")
            mac = _pypi_requirements(_ENVS / f"{engine}-macos.txt")
            with self.subTest(engine=engine, check="same packages"):
                self.assertEqual(set(cuda) - os_specific, set(mac) - os_specific)
            for name in sorted((set(cuda) & set(mac)) - os_specific):
                with self.subTest(engine=engine, package=name):
                    self.assertEqual(sorted(cuda[name]), sorted(mac[name]))


if __name__ == "__main__":
    unittest.main()
