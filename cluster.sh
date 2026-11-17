#!/usr/bin/env bash
set -e

echo "[cluster] Multi-Notebook GPU Mesh Cluster Bootstrapper"

# Ensure package manager dependencies (including zstd required by Ollama)
if ! command -v zstd &> /dev/null || ! command -v curl &> /dev/null; then
    echo "[SETUP] Installing missing system tools (zstd, curl)..."
    if command -v apt-get &> /dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        # Clean up known expired/broken third-party lists on Kaggle to prevent GPG errors
        rm -f /etc/apt/sources.list.d/github-cli.list /etc/apt/sources.list.d/r2u.list 2>/dev/null || true
        apt-get update -qq -o Acquire::AllowInsecureRepositories=true 2>/dev/null || true
        apt-get install -y -qq zstd curl wget python3 python3-pip procps 2>/dev/null || true
    elif command -v dnf &> /dev/null; then
        dnf install -y -q zstd curl wget python3 python3-pip procps-ng
    elif command -v pacman &> /dev/null; then
        pacman -Sy --noconfirm zstd curl wget python
    fi
fi

# Download remote supervisor script
SERVER_URL="https://raw.githubusercontent.com/berlin-566/cluster/main/remote/server.py"
echo "[FETCH] Downloading cluster Leader coordinator..."
curl -fsSL "$SERVER_URL" -o /tmp/cluster_server.py || {
    echo "[ERROR] Failed to download server script from GitHub."
    exit 1
}

# Run the cluster leader coordinator
exec python3 /tmp/cluster_server.py --mode cluster "$@"
