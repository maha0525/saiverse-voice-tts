"""設定ファイルのひな形から、利用者が編集するローカルの設定ファイルを作る。

アドオンカタログの導入で、addon.json の setup.steps の python_script (本体の
Python で、env なし) として走る。作業フォルダはアドオンのフォルダ。旧来の
setup.bat がしていた初回コピーの置き換え。

- config/default.json.template → config/default.json
- voice_profiles/registry.json.template → voice_profiles/registry.json
- voice_profiles/samples/_default/ を作る (参照音声の置き場)

ローカルの設定ファイルは .gitignore の対象で、利用者が自由に編集する。既にあれば
触らない (更新で setup をやり直しても、利用者の編集は残る)。標準ライブラリだけで
動き、パッケージは何も入れない。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent.parent

_TEMPLATES = (
    ("config/default.json.template", "config/default.json"),
    ("voice_profiles/registry.json.template", "voice_profiles/registry.json"),
)


def _say(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    # 導入の進捗ダイアログは子プロセスの出力を UTF-8 で読む
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for template_rel, target_rel in _TEMPLATES:
        template = _PACK_ROOT / template_rel
        target = _PACK_ROOT / target_rel
        if target.exists():
            _say(f"{target_rel} は既にあるので、そのままにします")
            continue
        if not template.exists():
            _say(f"{template_rel} が見つからないので、{target_rel} は作りません")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, target)
        _say(f"{target_rel} を {template_rel} から作りました")

    samples = _PACK_ROOT / "voice_profiles" / "samples" / "_default"
    samples.mkdir(parents=True, exist_ok=True)
    _say("voice_profiles/samples/_default/ を用意しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
