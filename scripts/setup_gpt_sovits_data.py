"""GPT-SoVITS のモデルの重みと NLTK のデータを取得する。

アドオンカタログの導入で、addon.json の setup.steps の python_script
(env: gpt_sovits) として、GPT-SoVITS 専用の Python 環境の Python で走る。作業
フォルダはアドオンのフォルダ。旧来の setup.bat / scripts/install_backends.py が
していた取得の置き換え。

- HuggingFace の lj1995/GPT-SoVITS → external/GPT-SoVITS/GPT_SoVITS/pretrained_models/
- external/GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/ を作る
  (fast_langdetect のモデルの置き場。GPT-SoVITS が初回に取得する)
- NLTK の averaged_perceptron_tagger_eng と cmudict → <専用環境>/nltk_data/

NLTK のデータは、保存先を指定しないと ~/nltk_data に落ちて、アンインストールで
消えない場所に残る。専用環境の下 (sys.prefix は専用環境のフォルダ) に置き、合成の
子プロセスも NLTK_DATA で同じ場所を指す (tools/speak/engine/subprocess_proxy.py)。
取得済みのものは取得し直さない (snapshot_download と nltk.download が既存を見る)。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parent.parent
_GPT_SOVITS = _PACK_ROOT / "external" / "GPT-SoVITS"
_PRETRAINED = _GPT_SOVITS / "GPT_SoVITS" / "pretrained_models"
_HF_REPO = "lj1995/GPT-SoVITS"
_NLTK_PACKAGES = ("averaged_perceptron_tagger_eng", "cmudict")


def _say(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    # 導入の進捗ダイアログは子プロセスの出力を UTF-8 で読む
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not _GPT_SOVITS.is_dir():
        _say(f"GPT-SoVITS が見つかりません: {_GPT_SOVITS}")
        return 1

    from huggingface_hub import snapshot_download

    _say(f"GPT-SoVITS のモデルの重み ({_HF_REPO}) を取得します (数 GB)")
    _PRETRAINED.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=_HF_REPO, local_dir=str(_PRETRAINED))
    _say(f"取得しました: {_PRETRAINED}")

    (_PRETRAINED / "fast_langdetect").mkdir(parents=True, exist_ok=True)

    import nltk

    nltk_dir = Path(sys.prefix) / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    for package in _NLTK_PACKAGES:
        _say(f"NLTK のデータ {package} を取得します")
        if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
            _say(f"NLTK のデータ {package} を取得できませんでした")
            return 1
    _say(f"NLTK のデータを置きました: {nltk_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
