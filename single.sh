#!/usr/bin/env bash
set -e

echo "[cluster] Cloud GPU Single Host Bootstrapper"

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
SERVER_URL="https://raw.githubusercontent.com/berlin-566/cluster/main/remote/server.py"
echo "[FETCH] Downloading cluster runner..."
curl -fsSL "$SERVER_URL" -o /tmp/cluster_server.py || {
    echo "[ERROR] Failed to download server script from GitHub."
    exit 1
}

# Run the single host runner
exec python3 /tmp/cluster_server.py --mode single "$@"
