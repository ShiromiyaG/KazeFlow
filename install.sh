#!/bin/bash
set -e

printf "\033]0;KazeFlow Installer\007"
clear

INSTALL_DIR="$(pwd)"
MINICONDA_DIR="$HOME/Miniconda3"
ENV_DIR="$INSTALL_DIR/env"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh"
CONDA_EXE="$MINICONDA_DIR/bin/conda"

SECONDS=0

log_message() {
    local msg="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $msg"
}

install_miniconda() {
    if [ -x "$CONDA_EXE" ]; then
        log_message "Miniconda already installed. Skipping."
        return
    fi

    log_message "Downloading Miniconda..."
    curl -fsSLo miniconda.sh "$MINICONDA_URL"

    if [ ! -f miniconda.sh ]; then
        log_message "Download failed. Check your internet connection."
        exit 1
    fi

    bash miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f miniconda.sh
    log_message "Miniconda installed."
}

create_conda_env() {
    if [ -d "$ENV_DIR" ]; then
        log_message "Conda environment already exists. Skipping creation."
        return
    fi

    log_message "Creating conda environment..."
    "$CONDA_EXE" create --no-shortcuts -y -k --prefix "$ENV_DIR" python=3.10.18

    if [ -x "$ENV_DIR/bin/python" ]; then
        log_message "Installing uv package installer..."
        "$ENV_DIR/bin/python" -m pip install uv
    fi
}

install_dependencies() {
    log_message "Installing dependencies..."

    # shellcheck disable=SC1091
    source "$MINICONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR"

    export UV_PYTHON="$ENV_DIR/bin/python"

    "$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools
    "$ENV_DIR/bin/python" -m pip install uv

    "$ENV_DIR/bin/uv" pip install --upgrade setuptools
    "$ENV_DIR/bin/uv" pip install torch torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu128
    "$ENV_DIR/bin/uv" pip install -r "$INSTALL_DIR/requirements.txt"

    unset UV_PYTHON
    conda deactivate

    log_message "Dependencies installed."
}

install_miniconda
create_conda_env
install_dependencies

elapsed=$SECONDS
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo ""
echo "Installation time: ${hours}h ${minutes}m ${seconds}s"
echo ""
echo "KazeFlow has been installed successfully!"
echo "Run ./start.sh to launch KazeFlow."
