#!/usr/bin/env bash
set -e

echo "[host-it] Cloud-to-Local LLM Bridge Bootstrapper"

# Ensure package manager dependencies (including zstd required by Ollama)
if command -v apt-get &> /dev/null; then
    echo "[SETUP] Installing system dependencies (zstd, curl, wget, python3)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq zstd curl wget python3 python3-pip procps
elif command -v dnf &> /dev/null; then
    echo "[SETUP] Installing system dependencies via dnf..."
    dnf install -y -q zstd curl wget python3 python3-pip procps-ng
elif command -v pacman &> /dev/null; then
    echo "[SETUP] Installing system dependencies via pacman..."
    pacman -Sy --noconfirm zstd curl wget python
fi

# Download remote supervisor script
SERVER_URL="https://raw.githubusercontent.com/berlin-566/host-it/main/remote/server.py"
echo "[FETCH] Downloading host-it runner..."
curl -fsSL "$SERVER_URL" -o /tmp/host_it_server.py || {
    echo "[ERROR] Failed to download server script from GitHub."
    exit 1
}

# Run the supervisor
python3 /tmp/host_it_server.py
