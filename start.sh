#!/bin/bash
set -e

if [ ! -d "env" ]; then
    echo "Please run './install.sh' first to set up the environment."
    read -rp "Press enter to exit..." _
    exit 1
fi

# Suppress PyTorch compile / Triton / inductor noise
export PYTHONWARNINGS="ignore"
unset TORCH_LOGS
export TORCHINDUCTOR_DISABLE_PROGRESS=1
export TORCHDYNAMO_VERBOSE=0
export TRITON_DISABLE_LINE_INFO=1
export TORCHINDUCTOR_VERBOSE=0
# Suppress GCC warnings from Triton JIT compilation
export CC="gcc"
export CXX="g++"
export CFLAGS="-w"
export CXXFLAGS="-w"

printf "\033]0;KazeFlow\007"
clear

env/bin/python app.py "$@"
