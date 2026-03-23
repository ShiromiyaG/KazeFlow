@echo off
title KazeFlow

if not exist env (
    echo Please run 'install.bat' first to set up the environment.
    pause
    exit /b 1
)

rem Suppress PyTorch compile / Triton / inductor noise
set PYTHONWARNINGS=ignore
set TORCH_LOGS=
set TORCHINDUCTOR_DISABLE_PROGRESS=1
set TORCHDYNAMO_VERBOSE=0

env\python.exe app.py %*
pause
