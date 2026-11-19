#!/usr/bin/env bash
set -e

echo "[cluster] Desktop Installer and Updater"

# Switch to home directory so broken working directories never block execution
cd "$HOME" || cd /tmp

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is required but was not found. Please install python3 first."
    exit 1
fi

INSTALL_DIR="$HOME/.local/share/cluster"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/.venv"
mkdir -p "$BIN_DIR"

# Clean up legacy host-it directory completely
if [ -d "$HOME/.local/share/host-it" ]; then
    echo "[CLEANUP] Removing legacy host-it directory..."
    rm -rf "$HOME/.local/share/host-it"
fi

# Channel selection: Main branch vs Latest release
INSTALL_CHANNEL="${CLUSTER_CHANNEL:-}"
if [ -z "$INSTALL_CHANNEL" ]; then
    INSTALL_CHANNEL="main"
    if [ -t 0 ]; then
        echo ""
        echo "Select installation source channel:"
        echo "  1) Main branch (Recommended: latest features and fixes)"
        echo "  2) Latest release (Stable release tags)"
        read -r -t 15 -p "Enter choice [1/2, default 1 in 15s]: " USER_CHOICE || USER_CHOICE="1"
        case "$USER_CHOICE" in
            2) INSTALL_CHANNEL="release" ;;
            *) INSTALL_CHANNEL="main" ;;
        esac
    elif [ -c /dev/tty ]; then
        echo ""
        echo "Select installation source channel:"
        echo "  1) Main branch (Recommended: latest features and fixes)"
        echo "  2) Latest release (Stable release tags)"
        read -r -t 15 -p "Enter choice [1/2, default 1 in 15s]: " USER_CHOICE < /dev/tty 2>/dev/null || USER_CHOICE="1"
        case "$USER_CHOICE" in
            2) INSTALL_CHANNEL="release" ;;
            *) INSTALL_CHANNEL="main" ;;
        esac
    fi
fi

IS_UPDATE=0
if [ -d "$INSTALL_DIR" ] && [ -f "$BIN_DIR/cluster" ]; then
    IS_UPDATE=1
    echo "[CHECK] Existing cluster installation detected. Updating via $INSTALL_CHANNEL channel..."
else
    echo "[CHECK] Performing fresh cluster installation via $INSTALL_CHANNEL channel..."
fi

echo "[SETUP] Fetching cluster release from GitHub..."
TARGET_REF="main"

if command -v git &> /dev/null; then
    if [ ! -d "$INSTALL_DIR/.git" ]; then
        rm -rf "$INSTALL_DIR"
        git clone https://github.com/berlin-566/cluster.git "$INSTALL_DIR"
    else
        git -C "$INSTALL_DIR" remote set-url origin https://github.com/berlin-566/cluster.git 2>/dev/null || true
        git -C "$INSTALL_DIR" fetch origin
    fi

    if [ "$INSTALL_CHANNEL" = "release" ]; then
        LATEST_TAG=$(git -C "$INSTALL_DIR" tag -l --sort=-v:refname | head -n 1)
        if [ -n "$LATEST_TAG" ]; then
            TARGET_REF="$LATEST_TAG"
            echo "[CHANNEL] Checking out latest release tag: $LATEST_TAG"
            git -C "$INSTALL_DIR" checkout -q "$LATEST_TAG"
        else
            echo "[CHANNEL] No release tags published yet. Using main branch."
            TARGET_REF="main"
            git -C "$INSTALL_DIR" checkout -q main
            git -C "$INSTALL_DIR" reset -q --hard origin/main
        fi
    else
        TARGET_REF="main"
        git -C "$INSTALL_DIR" checkout -q main
        git -C "$INSTALL_DIR" reset -q --hard origin/main
    fi
else
    mkdir -p "$INSTALL_DIR"
    curl -fsSL https://github.com/berlin-566/cluster/archive/refs/heads/main.tar.gz | tar -xz -C "$INSTALL_DIR" --strip-components=1
fi

# Rebuild virtual environment if missing or if python3/pip interpreter is broken (e.g. from relocated dir)
if [ -d "$VENV_DIR" ]; then
    if ! "$VENV_DIR/bin/python3" -c "import sys" &> /dev/null || ! "$VENV_DIR/bin/pip" --version &> /dev/null; then
        echo "[SETUP] Rebuilding virtual environment to ensure valid interpreter paths..."
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Creating dedicated virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[SETUP] Updating dependencies and CLI package..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -e "$INSTALL_DIR"

# Record install metadata
COMMIT_HASH="unknown"
if command -v git &> /dev/null && [ -d "$INSTALL_DIR/.git" ]; then
    COMMIT_HASH=$(git -C "$INSTALL_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
fi

cat << EOF > "$INSTALL_DIR/.install_info"
{
  "channel": "$INSTALL_CHANNEL branch",
  "target_ref": "$TARGET_REF",
  "commit": "$COMMIT_HASH",
  "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# Ensure executable launcher exists in ~/.local/bin/cluster
cat << 'EOF' > "$BIN_DIR/cluster"
#!/usr/bin/env bash
INSTALL_DIR="$HOME/.local/share/cluster"
exec "$INSTALL_DIR/.venv/bin/python" -m cluster "$@"
EOF
chmod +x "$BIN_DIR/cluster"

# Keep backward compatibility launcher for host-it
cat << 'EOF' > "$BIN_DIR/host-it"
#!/usr/bin/env bash
INSTALL_DIR="$HOME/.local/share/cluster"
exec "$INSTALL_DIR/.venv/bin/python" -m cluster "$@"
EOF
chmod +x "$BIN_DIR/host-it"

# Check if ~/.local/bin is in PATH
PATH_UPDATED=0
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        SHELL_RC=""
        if [ -n "$BASH_VERSION" ]; then
            SHELL_RC="$HOME/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
            if ! grep -q 'PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC" 2>/dev/null; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
                PATH_UPDATED=1
            fi
        fi
        export PATH="$BIN_DIR:$PATH"
        ;;
esac

if [ "$IS_UPDATE" -eq 1 ]; then
    echo "[OK] cluster has been updated to the latest version successfully!"
else
    echo "[OK] cluster installed successfully!"
fi

echo "Launcher:        $BIN_DIR/cluster (alias: $BIN_DIR/host-it)"
echo "Channel:         $INSTALL_CHANNEL branch ($COMMIT_HASH)"
if [ "$PATH_UPDATED" -eq 1 ]; then
    echo "[NOTE] Added $BIN_DIR to your PATH. If 'cluster' is not recognized in current terminal, run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo ""
echo "Verify installation:"
echo "  cluster --version"
echo ""
echo "You can now run:"
echo "  cluster admin connect <tunnel-url> --token <admin-token>"
echo "  cluster connect <tunnel-url> --token <token>"
echo "  cluster ui                 (Launch Terminal Cockpit Dashboard)"
echo "  cluster fetch <model>      (Search & Deploy Models)"
echo "  cluster serve              (Start Local API Server & Ollama Bridge on 127.0.0.1:8080)
  cluster proxy              (Alias for cluster serve)"
echo "  cluster chat               (Terminal Streaming Chat)"
