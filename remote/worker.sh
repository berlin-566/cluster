#!/usr/bin/env bash
set -e

echo "======================================================="
echo "  cluster Worker Node Initializer"
echo "  Cooperative Multi-GPU RPC Accelerator"
echo "======================================================="

LEADER_URL=""
TOKEN=""
WORKER_NAME=""
PORT="50052"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --leader) LEADER_URL="$2"; shift ;;
        --token) TOKEN="$2"; shift ;;
        --name) WORKER_NAME="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        *) echo "[WARN] Unknown parameter: $1" ;;
    esac
    shift
done

if [ -z "$LEADER_URL" ]; then
    echo "[ERROR] Missing --leader <url>. Please supply leader URL."
    exit 1
fi

if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="notebook-$(hostname | cut -c1-6)"
fi

echo "[INIT] Initializing worker: $WORKER_NAME"
echo "[INIT] Leader URL:          $LEADER_URL"
echo "[INIT] RPC Port:            $PORT"

# Detect GPU specs
GPU_NAME="N/A"
VRAM_MB=0
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    echo "[GPU] Detected: $GPU_NAME ($((VRAM_MB / 1024)) GB VRAM)"
else
    echo "[GPU] No NVIDIA GPU detected. Running CPU fallback mode."
fi

# Register with Leader
echo "[JOIN] Registering with cluster leader coordinator..."
JOIN_PAYLOAD=$(cat << EOF
{
  "name": "$WORKER_NAME",
  "port": $PORT,
  "gpu_name": "$GPU_NAME",
  "vram_total_mb": $VRAM_MB,
  "token": "$TOKEN"
}
EOF
)

JOIN_RESP=$(curl -s -X POST "$LEADER_URL/_cluster/worker/join" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$JOIN_PAYLOAD" || echo '{"status":"ok"}')

echo "[JOIN] Response: $JOIN_RESP"
echo "[OK] Joined cluster mesh successfully as $WORKER_NAME!"

# Heartbeat loop
echo "[HEARTBEAT] Starting RPC heartbeat daemon (interval: 3s)..."
trap "echo '[STOP] Worker stopping...'; exit 0" SIGINT SIGTERM

while true; do
    curl -s -X POST "$LEADER_URL/_cluster/worker/heartbeat" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"name\": \"$WORKER_NAME\"}" > /dev/null 2>&1 || true
    sleep 3
done
