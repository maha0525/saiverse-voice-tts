#!/usr/bin/env bash
# ===========================================================================
#  Voice TTS speech synthesis server (Linux / macOS)
#
#  Usage:
#    ./serve.sh                              - GPT-SoVITS on 127.0.0.1:8082
#    ./serve.sh --host 0.0.0.0               - reachable from other PCs (no auth;
#                                              use only inside a LAN / Tailscale)
#    ./serve.sh --engine irodori --port 8083
#    ./serve.sh --python /path/to/python --data-dir /path/to/data
#                                            - without SAIVerse: run with a Python
#                                              that has the engine installed
#
#  Choose the GPU with CUDA_VISIBLE_DEVICES, e.g.
#    CUDA_VISIBLE_DEVICES=1 ./serve.sh
#
#  The launcher (tools/speak/serve_launcher.py) runs with SAIVerse's Python
#  (two folders above this one) and starts the server with the engine's own
#  Python environment. It stops with a reason when that environment is missing.
# ===========================================================================
set -u

PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAIVERSE_ROOT="$(cd "$PACK_ROOT/../.." && pwd)"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# The Python given with --python (if any): it can also run the launcher, which
# needs only the standard library in --python mode.
ARG_PY=""
prev=""
for arg in "$@"; do
    case "$arg" in
        --python=*) ARG_PY="${arg#--python=}" ;;
        *) [ "$prev" = "--python" ] && ARG_PY="$arg" ;;
    esac
    prev="$arg"
done

LAUNCHER_PY="$SAIVERSE_ROOT/.venv/bin/python"
if [ ! -x "$LAUNCHER_PY" ]; then
    echo "[voice-tts] SAIVerse's Python was not found at $LAUNCHER_PY." >&2
    echo "[voice-tts] Without SAIVerse only --python mode works." >&2
    if [ -n "$ARG_PY" ]; then
        LAUNCHER_PY="$ARG_PY"
    else
        LAUNCHER_PY="$(command -v python3 || command -v python || true)"
    fi
    if [ -z "$LAUNCHER_PY" ]; then
        echo "[voice-tts] No python found (give one with --python)." >&2
        exit 2
    fi
fi

# --saiverse-root: the same SAIVerse folder this script took the Python from
# (a later --saiverse-root in the arguments wins)
exec "$LAUNCHER_PY" "$PACK_ROOT/tools/speak/serve_launcher.py" --saiverse-root "$SAIVERSE_ROOT" "$@"
