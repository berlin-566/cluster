#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  🚀 host-it: Cloud-to-Local LLM Bridge Bootstrapper"
echo "=========================================================="

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Installing python3..."
    apt-get update -qq && apt-get install -y -qq python3 python3-pip curl wget
fi

# Download remote supervisor script
SERVER_URL="https://raw.githubusercontent.com/berlin-566/host-it/main/remote/server.py"
echo "📥 Fetching host-it runner..."
curl -fsSL "$SERVER_URL" -o /tmp/host_it_server.py || {
    echo "❌ Failed to download server script from GitHub. Trying fallback..."
}

# Run the supervisor
python3 /tmp/host_it_server.py
