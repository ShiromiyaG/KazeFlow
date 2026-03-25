#!/bin/bash
set -e

export PYTHONWARNINGS="ignore"
export TF_CPP_MIN_LOG_LEVEL="3"

host="127.0.0.1"
port=25565
address="http://$host:$port"

echo "TensorBoard Address: $address"

read -rp "Key in model saved directory: " UserInputPath

if [ -z "$UserInputPath" ]; then
    echo "No directory provided. Exiting."
    exit 1
fi

tensorboard --logdir="$UserInputPath" --host="$host" --port="$port" \
    --samples_per_plugin images=9999,audio=9999 &

TB_PID=$!

sleep 3

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$address" >/dev/null 2>&1
elif command -v open >/dev/null 2>&1; then
    open "$address" >/dev/null 2>&1
else
    echo "Could not detect web browser. Please open $address manually."
fi

echo "TensorBoard is running with PID $TB_PID."
echo "Press Ctrl+C to stop TensorBoard and exit."

trap "kill $TB_PID 2>/dev/null" EXIT

wait $TB_PID