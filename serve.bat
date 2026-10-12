@echo off
REM ===========================================================================
REM  Voice TTS speech synthesis server (Windows)
REM
REM  Usage:
REM    serve.bat                              - GPT-SoVITS on 127.0.0.1:8082
REM    serve.bat --host 0.0.0.0               - reachable from other PCs (no auth;
REM                                             use only inside a LAN / Tailscale)
REM    serve.bat --engine irodori --port 8083
REM    serve.bat --python C:\path\to\python.exe --data-dir C:\path\to\data
REM                                           - without SAIVerse: run with a Python
REM                                             that has the engine installed
REM
REM  Choose the GPU with CUDA_VISIBLE_DEVICES before running, e.g.
REM    set CUDA_VISIBLE_DEVICES=1
REM
REM  The launcher (tools\speak\serve_launcher.py) runs with SAIVerse's Python
REM  (two folders above this one) and starts the server with the engine's own
REM  Python environment. It stops with a reason when that environment is missing.
REM ===========================================================================
setlocal
set "PACK_ROOT=%~dp0"
if "%PACK_ROOT:~-1%"=="\" set "PACK_ROOT=%PACK_ROOT:~0,-1%"
for %%I in ("%PACK_ROOT%\..\..") do set "SAIVERSE_ROOT=%%~fI"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "LAUNCHER_PY=%SAIVERSE_ROOT%\.venv\Scripts\python.exe"
if exist "%LAUNCHER_PY%" goto run

echo [voice-tts] SAIVerse's Python was not found at "%LAUNCHER_PY%". 1>&2
echo [voice-tts] Running the launcher with python on PATH; without SAIVerse only --python mode works. 1>&2
set "LAUNCHER_PY="
for /f "delims=" %%P in ('where python 2^>nul') do if not defined LAUNCHER_PY set "LAUNCHER_PY=%%P"
if not defined LAUNCHER_PY (
    echo [voice-tts] No python found on PATH. 1>&2
    exit /b 2
)

:run
REM --saiverse-root: the same SAIVerse folder this script took the Python from
REM (a later --saiverse-root in the arguments wins)
"%LAUNCHER_PY%" "%PACK_ROOT%\tools\speak\serve_launcher.py" --saiverse-root "%SAIVERSE_ROOT%" %*
exit /b %ERRORLEVEL%
