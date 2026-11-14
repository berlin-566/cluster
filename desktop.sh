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

# Migrate old host-it install directory if present
if [ -d "$HOME/.local/share/host-it" ] && [ ! -d "$INSTALL_DIR" ]; then
    echo "[MIGRATE] Migrating legacy ~/.local/share/host-it to $INSTALL_DIR..."
    mv "$HOME/.local/share/host-it" "$INSTALL_DIR"
fi

IS_UPDATE=0
if [ -d "$INSTALL_DIR" ] && [ -f "$BIN_DIR/cluster" ]; then
    IS_UPDATE=1
    echo "[CHECK] Existing cluster installation detected. Updating..."
else
    echo "[CHECK] Performing fresh cluster installation..."
fi

echo "[SETUP] Fetching latest cluster release..."
if command -v git &> /dev/null; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        git -C "$INSTALL_DIR" remote set-url origin https://github.com/berlin-566/cluster.git 2>/dev/null || true
        git -C "$INSTALL_DIR" fetch --depth 1 origin main
        git -C "$INSTALL_DIR" reset --hard origin/main
    else
        rm -rf "$INSTALL_DIR"
        git clone --depth 1 https://github.com/berlin-566/cluster.git "$INSTALL_DIR"
    fi
else
    mkdir -p "$INSTALL_DIR"
    curl -fsSL https://github.com/berlin-566/cluster/archive/refs/heads/main.tar.gz | tar -xz -C "$INSTALL_DIR" --strip-components=1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Creating dedicated virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[SETUP] Updating dependencies and CLI package..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -e "$INSTALL_DIR"

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

echo "Launcher: $BIN_DIR/cluster (alias: $BIN_DIR/host-it)"
if [ "$PATH_UPDATED" -eq 1 ]; then
    echo "[NOTE] Added $BIN_DIR to your PATH. If 'cluster' is not recognized in current terminal, run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo ""
echo "You can now run:"
echo "  cluster connect <tunnel-url> --token <token>"
echo "  cluster proxy"
echo "  cluster chat"
