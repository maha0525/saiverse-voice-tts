"""Irodori-TTS のモデルの重みを取得する。

アドオンカタログの導入で、addon.json の setup.steps の python_script
(env: irodori) として、Irodori-TTS 専用の Python 環境の Python で走る。

- Aratako/Irodori-TTS-500M-v2 (モデル)
- Aratako/Semantic-DACVAE-Japanese-32dim (コーデック)

保存先は HuggingFace のキャッシュ (local_dir を指定しない)。合成のときに
Irodori-TTS がキャッシュから読むので、初回の合成でダウンロードを待たずに済む。
キャッシュの場所 (HF_HOME) は専用環境の下 (<専用環境>/hf_cache/) に固定する。
合成の子プロセスも HF_HOME で同じ場所を指し (tools/speak/engine/subprocess_proxy.py)、
アンインストールで専用環境と一緒に消える。利用者の環境変数に HF_HOME があっても
上書きする — そちらに取得すると、合成の子プロセスが見る場所と食い違い、初回の合成で
もう一度ダウンロードが走る。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# huggingface_hub は import の時点で HF_HOME を読むので、import より前に置く
os.environ["HF_HOME"] = str(Path(sys.prefix) / "hf_cache")

_HF_REPOS = (
    "Aratako/Irodori-TTS-500M-v2",
    "Aratako/Semantic-DACVAE-Japanese-32dim",
)


def _say(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    # 導入の進捗ダイアログは子プロセスの出力を UTF-8 で読む
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from huggingface_hub import snapshot_download

    for repo_id in _HF_REPOS:
        _say(f"Irodori-TTS のモデルの重み ({repo_id}) を取得します")
        path = snapshot_download(repo_id=repo_id)
        _say(f"取得しました: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
