"""取得した Irodori-TTS の依存の版の縛りを、導入できる形に手直しする。

アドオンカタログの導入で、Irodori-TTS の git_clone の直後 (pip_install の前) に
python_script (env なし、標準ライブラリだけ) として走る。

手直しは一つ: ``sentencepiece>=0.1.99,<0.2`` → ``sentencepiece>=0.1.99,<0.3``。
0.2 未満の最後の版 (0.1.99) には Python 3.13 の Windows 用 wheel が無く、
ソースビルドも今の CMake (4 系) が古いビルド設定を拒んで失敗する。0.2 系には
wheel があり、import の API は同じ (隔離の通し確認 2026-10-05 で発見。上流の
固定 commit 89f9d8f の pyproject.toml が対象で、commit を上げるときはこの
手直しが要るかを見直す)。

対象の文字列が見つからなければ失敗で止める — 黙って素通りすると、commit を
上げたときに縛りが戻っても気づけない。二度目の実行 (手直し済み) は成功扱い。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PYPROJECT = Path("external/Irodori-TTS/pyproject.toml")
_OLD = '"sentencepiece>=0.1.99,<0.2"'
_NEW = '"sentencepiece>=0.1.99,<0.3"'


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not _PYPROJECT.exists():
        print(f"{_PYPROJECT} が見つかりません (git_clone の step が先に要る)", flush=True)
        return 1
    text = _PYPROJECT.read_text(encoding="utf-8")
    if _NEW in text:
        print("sentencepiece の縛りは手直し済みです", flush=True)
        return 0
    if _OLD not in text:
        print(
            f"想定している縛り {_OLD} が pyproject.toml に見つかりません。"
            "Irodori-TTS の commit を上げたときは、この手直しが今も要るかを見直してください",
            flush=True,
        )
        return 1
    _PYPROJECT.write_text(text.replace(_OLD, _NEW), encoding="utf-8")
    print(f"sentencepiece の縛りを手直ししました: {_OLD} -> {_NEW}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
