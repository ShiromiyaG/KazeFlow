#!/bin/bash
set -e

if [ ! -d "env" ]; then
    echo "Please run './install.sh' first to set up the environment."
    read -rp "Press enter to exit..." _
    exit 1
fi

printf "\033]0;KazeFlow\007"
clear

env/bin/python app.py "$@"
