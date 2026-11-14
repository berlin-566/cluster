#!/usr/bin/env bash
set -e

echo "[host-it] Desktop Installer and Updater"

# Switch to home directory so broken working directories never block execution
cd "$HOME" || cd /tmp

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 is required but was not found. Please install python3 first."
    exit 1
fi

INSTALL_DIR="$HOME/.local/share/host-it"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/.venv"
mkdir -p "$BIN_DIR"

IS_UPDATE=0
if [ -d "$INSTALL_DIR" ] && [ -f "$BIN_DIR/host-it" ]; then
    IS_UPDATE=1
    echo "[CHECK] Existing host-it installation detected. Updating..."
else
    echo "[CHECK] Performing fresh host-it installation..."
fi

echo "[SETUP] Fetching latest host-it release..."
if command -v git &> /dev/null; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        git -C "$INSTALL_DIR" fetch --depth 1 origin main
        git -C "$INSTALL_DIR" reset --hard origin/main
    else
        rm -rf "$INSTALL_DIR"
        git clone --depth 1 https://github.com/berlin-566/host-it.git "$INSTALL_DIR"
    fi
else
    mkdir -p "$INSTALL_DIR"
    curl -fsSL https://github.com/berlin-566/host-it/archive/refs/heads/main.tar.gz | tar -xz -C "$INSTALL_DIR" --strip-components=1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Creating dedicated virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[SETUP] Updating dependencies and CLI package..."
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -e "$INSTALL_DIR"

# Ensure executable launcher exists in ~/.local/bin/host-it
cat << 'EOF' > "$BIN_DIR/host-it"
#!/usr/bin/env bash
INSTALL_DIR="$HOME/.local/share/host-it"
exec "$INSTALL_DIR/.venv/bin/python" -m host_it "$@"
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
    echo "[OK] host-it has been updated to the latest version successfully!"
else
    echo "[OK] host-it installed successfully!"
fi

echo "Launcher: $BIN_DIR/host-it"
if [ "$PATH_UPDATED" -eq 1 ]; then
    echo "[NOTE] Added $BIN_DIR to your PATH. If 'host-it' is not recognized in current terminal, run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo ""
echo "You can now run:"
echo "  host-it connect <tunnel-url> --token <token>"
echo "  host-it proxy"
echo "  host-it chat"
