@echo off
title KazeFlow

if not exist env (
    echo Please run 'install.bat' first to set up the environment.
    pause
    exit /b 1
)

rem Force UTF-8 for Python I/O — prevents UnicodeEncodeError in torch.compile generated code
set PYTHONUTF8=1

rem Add FFmpeg DLLs and MinGW gcc to PATH so torchcodec and Triton can find them
set "PATH=%~dp0env\Library\bin;%PATH%"

rem Suppress PyTorch compile / Triton / inductor noise
set PYTHONWARNINGS=ignore
set TORCH_LOGS=
set TORCHINDUCTOR_DISABLE_PROGRESS=1
set TORCHDYNAMO_VERBOSE=0
set TORCHINDUCTOR_VERBOSE=0
set TRITON_DISABLE_LINE_INFO=1

rem Point Triton to the MinGW gcc in our env
set CC=%~dp0env\Library\bin\gcc.exe
set CFLAGS=-w

env\python.exe app.py %*
pause
