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

# Clean trailing slashes from LEADER_URL and ensure protocol
LEADER_URL="${LEADER_URL%/}"
if [[ "$LEADER_URL" != http* ]]; then
    LEADER_URL="https://$LEADER_URL"
fi

if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="notebook-$(hostname 2>/dev/null | cut -c1-6 || echo "$RANDOM")"
fi

echo "[INIT] Initializing worker: $WORKER_NAME"
echo "[INIT] Leader URL:          $LEADER_URL"
echo "[INIT] RPC Port:            $PORT"

# Detect GPU specs
GPU_NAME="Tesla T4"
VRAM_MB=16384
if command -v nvidia-smi &> /dev/null; then
    DETECTED_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    DETECTED_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    if [ -n "$DETECTED_NAME" ]; then
        GPU_NAME="$DETECTED_NAME"
    fi
    if [ -n "$DETECTED_VRAM" ]; then
        VRAM_MB="$DETECTED_VRAM"
    fi
    echo "[GPU] Detected: $GPU_NAME ($((VRAM_MB / 1024)) GB VRAM)"
else
    echo "[GPU] No NVIDIA GPU detected. Running simulated T4 accelerator mode."
fi

# Register with Leader (retry up to 15 times with 2s interval)
echo "[JOIN] Registering with cluster leader coordinator at $LEADER_URL..."
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

JOINED=0
for i in {1..15}; do
    JOIN_RESP=$(curl -s --connect-timeout 5 --max-time 10 -X POST "$LEADER_URL/_cluster/worker/join" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "$JOIN_PAYLOAD" 2>/dev/null || true)

    if echo "$JOIN_RESP" | grep -q "joined"; then
        echo "[JOIN] Handshake response: $JOIN_RESP"
        echo "[OK] Joined cluster mesh successfully as $WORKER_NAME!"
        JOINED=1
        break
    else
        echo "[JOIN] Waiting for Leader endpoint ($LEADER_URL) to respond (attempt $i/15)..."
        sleep 2
    fi
done

if [ "$JOINED" -ne 1 ]; then
    echo "[WARN] Initial join handshake timed out, continuing via heartbeat loop..."
fi

# Heartbeat loop
echo "[HEARTBEAT] Starting RPC heartbeat daemon (interval: 3s)..."
trap "echo '[STOP] Worker stopping...'; exit 0" SIGINT SIGTERM

FAIL_COUNT=0
while true; do
    HB_RESP=$(curl -s --connect-timeout 4 --max-time 6 -X POST "$LEADER_URL/_cluster/worker/heartbeat" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $TOKEN" \
        -d "{\"name\": \"$WORKER_NAME\", \"gpu_name\": \"$GPU_NAME\", \"vram_total_mb\": $VRAM_MB}" 2>/dev/null || true)

    if echo "$HB_RESP" | grep -q "ok"; then
        FAIL_COUNT=0
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$FAIL_COUNT" -eq 10 ]; then
            echo "[HEARTBEAT] Warning: Leader missed 10 consecutive heartbeats, retrying..."
            FAIL_COUNT=0
        fi
    fi
    sleep 3
done
