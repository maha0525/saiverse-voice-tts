"""音声合成サーバー (server.py) の起動役。serve.bat / serve.sh から呼ばれる。

動き方は二通り (docs/intent/synthesis_server.md「サーバーの起動」):

- 既定 (SAIVerse に Voice TTS を入れた機械): SAIVerse の Python で動く。SAIVerse の
  フォルダ (Voice TTS のフォルダから二つ上) を読み込み先に足し、子プロセスと同じ
  方法 (subprocess_proxy.resolve_worker_python) でエンジン専用の Python 環境を探す。
  見つからなければ、いまの Python で黙って動かさず、理由を出して止まる。サーバー
  には子プロセスと同じ環境変数 (NLTK_DATA、HF_HOME など) を渡す。
- ``--python <path>``: SAIVerse を探さず、指定した Python と呼び出し側の環境変数の
  ままサーバーを動かす。``--data-dir`` が必須。

``--engine`` / ``--data-dir`` 以外の引数 (``--host`` / ``--port`` / ``--config``) は
そのまま server.py に渡す。サーバーの終了コードをそのまま返す。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SPEAK_DIR = Path(__file__).resolve().parent
_PACK_ROOT = _SPEAK_DIR.parent.parent
SERVER_PATH = _SPEAK_DIR / "server.py"
# アドオンは <SAIVerse>/expansion_data/saiverse-voice-tts/ に入る
DEFAULT_SAIVERSE_ROOT = _PACK_ROOT.parent.parent
ADDON_ID = "saiverse-voice-tts"


class LaunchError(RuntimeError):
    """サーバーを起動できない理由 (利用者に見せる文)。"""


def parse_args(argv: Sequence[str]) -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(
        description=(
            "Start the Voice TTS speech synthesis server. Other options "
            "(--host, --port, --config) are passed to the server."
        ),
    )
    ap.add_argument("--engine", default="gpt_sovits", help="gpt_sovits (default) or irodori")
    ap.add_argument("--data-dir", default=None,
                    help="where registered profiles are stored (required with --python)")
    ap.add_argument("--python", default=None,
                    help="run the server with this Python and the current environment "
                         "instead of SAIVerse's engine environment")
    ap.add_argument("--saiverse-root", default=None,
                    help="SAIVerse folder (default: two levels above the Voice TTS folder)")
    return ap.parse_known_args(list(argv))


def _import_engine_helpers(saiverse_root: Path):
    """SAIVerse のフォルダと tools/speak を読み込み先に足して、探し方を import する。"""
    root = str(saiverse_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    # engine パッケージは tools/speak から (SAIVerse 側の名前より先に解決させる)
    speak = str(_SPEAK_DIR)
    if speak in sys.path:
        sys.path.remove(speak)
    sys.path.insert(0, speak)
    from engine import subprocess_proxy

    return subprocess_proxy


def resolve_saiverse_engine(
    engine_name: str, saiverse_root: Path,
) -> Tuple[str, Dict[str, str], Path]:
    """SAIVerse に入れた機械: (エンジン専用の Python, サーバーに渡す環境変数, 既定の置き場所)。"""
    proxy = _import_engine_helpers(saiverse_root)
    if engine_name not in proxy.WORKER_ENGINES:
        raise LaunchError(
            f"unknown engine {engine_name!r} (choose from: {', '.join(proxy.WORKER_ENGINES)})"
        )
    env_name = proxy.WORKER_ENGINES[engine_name][0]
    try:
        python, env = proxy.resolve_worker_python(env_name, strict=True)
    except proxy.WorkerEnvError as exc:
        raise LaunchError(str(exc)) from exc
    if env is None:  # strict では起きない。起きたら黙って進まない
        raise LaunchError(f"engine environment for {engine_name} could not be resolved")
    from saiverse.addon_paths import get_addon_data_dir  # type: ignore  # 上で import 済み

    data_dir = get_addon_data_dir(ADDON_ID) / "server"
    return python, env, data_dir


def build_command(
    python: str, engine_name: str, data_dir: Path, passthrough: Sequence[str],
) -> List[str]:
    return [
        str(python), str(SERVER_PATH),
        "--engine", engine_name,
        "--data-dir", str(Path(data_dir).expanduser().resolve()),
        *passthrough,
    ]


def plan(argv: Sequence[str]) -> Tuple[List[str], Optional[Dict[str, str]]]:
    """起動するコマンドと環境変数を決める (None = 呼び出し側の環境のまま)。"""
    args, passthrough = parse_args(argv)
    if args.python:
        if not args.data_dir:
            raise LaunchError("--data-dir is required with --python")
        python = Path(args.python).expanduser()
        if not python.exists():
            raise LaunchError(f"--python {python} does not exist")
        return build_command(str(python), args.engine, Path(args.data_dir), passthrough), None

    root = Path(args.saiverse_root).expanduser() if args.saiverse_root else DEFAULT_SAIVERSE_ROOT
    python, env, default_data_dir = resolve_saiverse_engine(args.engine, root)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir
    return build_command(python, args.engine, data_dir, passthrough), env


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        command, env = plan(sys.argv[1:] if argv is None else argv)
    except LaunchError as exc:
        print(f"voice-tts server: {exc}", file=sys.stderr)
        return 2
    print(f"voice-tts server: starting {' '.join(command)}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(command, env=env)
    interrupts = 0
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # Ctrl+C はサーバーにも届いている。サーバーの終了を待って、その終了
            # コードを返す。二度目の Ctrl+C ではサーバーを止める
            interrupts += 1
            if interrupts >= 2:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
