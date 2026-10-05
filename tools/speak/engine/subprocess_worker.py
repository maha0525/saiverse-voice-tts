"""音声合成子プロセス本体。

本体プロセス (SAIVerse) とは独立した素の Python プロセスとして起動され、
``--engine`` で指定されたローカル合成エンジン (``GPTSoVITSEngine`` /
``IrodoriEngine``) の本物を保持する。標準入力でリクエストを受け、標準出力に
音声フレームを流す (プロトコルは subprocess_ipc を参照)。

アドオンカタログで導入した場合は、そのエンジン専用の Python 環境の Python で
起動される (起動する側: subprocess_proxy.resolve_worker_python)。

独立プロセスなので本体プロセスの GIL 競合の影響を受けない。また、本体の
``tools/`` パッケージを sys.path に入れないため、GPT-SoVITS 内部の
``from tools.X import`` は GPT-SoVITS 自身の ``tools/`` に解決され、本体側で
必要だった addon_external_loader の名前空間リダイレクトは不要になる。

起動: ``python subprocess_worker.py --engine gpt_sovits|irodori --config '<json>'``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# .../tools/speak を import 経路に追加し、engine パッケージを import 可能にする。
# 本体 (SAIVerse root) や addon の tools 親は **入れない** — それが tools 衝突を
# 避ける肝。
_SPEAK_DIR = Path(__file__).resolve().parent.parent
if str(_SPEAK_DIR) not in sys.path:
    sys.path.insert(0, str(_SPEAK_DIR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("gpt_sovits", "irodori"), default="gpt_sovits")
    ap.add_argument("--config", default="{}")
    args = ap.parse_args()

    try:
        config = json.loads(args.config)
    except Exception:
        config = {}

    # 標準出力をバイナリ通信路として確保し、ライブラリの print / tqdm は
    # stderr に逃がす (GPT-SoVITS は stdout に大量に print するため、これを
    # しないとプロトコルが壊れる。Irodori-TTS も同じ扱い)。stderr は親が errlog ファイルに転送する。
    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    proto_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    sys.stdout = sys.stderr  # type: ignore[assignment]

    from engine.subprocess_ipc import (
        decode_request,
        encode_chunk,
        encode_end,
        encode_error,
        encode_result,
        read_frame,
        write_frame,
    )

    if args.engine == "irodori":
        from engine.irodori import IrodoriEngine

        engine = IrodoriEngine(config)
    else:
        from engine.gpt_sovits import GPTSoVITSEngine

        engine = GPTSoVITSEngine(config)
    print(
        f"voice-tts worker: started (engine={args.engine}, python={sys.executable})",
        file=sys.stderr, flush=True,
    )

    while True:
        payload = read_frame(proto_in)
        if payload is None:
            break  # 親プロセスが stdin を閉じた = 終了

        try:
            req = decode_request(payload)
            method = req.get("method", "stream")
            text = req.get("text", "")
            ref_audio = req.get("ref_audio")
            ref_text = req.get("ref_text")
            params = req.get("params") or {}

            if method == "synthesize":
                result = engine.synthesize(text, ref_audio, ref_text, params)
                write_frame(
                    proto_out,
                    encode_result(result.audio, result.sample_rate, result.duration_ms),
                )
            else:
                for chunk in engine.synthesize_stream(text, ref_audio, ref_text, params):
                    write_frame(proto_out, encode_chunk(chunk.audio, chunk.sample_rate))
            write_frame(proto_out, encode_end())
        except Exception as exc:  # 1 リクエストの失敗でプロセスは落とさない
            import traceback

            traceback.print_exc(file=sys.stderr)
            try:
                write_frame(proto_out, encode_error(f"{type(exc).__name__}: {exc}"))
                write_frame(proto_out, encode_end())
            except Exception:
                break  # 出力路が壊れていたら諦めて終了

    return 0


if __name__ == "__main__":
    sys.exit(main())
