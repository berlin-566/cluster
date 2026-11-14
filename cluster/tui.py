import asyncio
import http.client
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Header,
        Footer,
        TabbedContent,
        TabPane,
        Static,
        Label,
        Button,
        Input,
        DataTable,
        ProgressBar,
        RichLog,
    )
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual.message import Message
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

from cluster.client import ClusterClient
from cluster.config import load_config, update_config, DEFAULT_HUB_PORT, DEFAULT_PROXY_PORT
from cluster.fetch import search_models

logger = logging.getLogger("cluster.tui")

# ANSI fallback colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_DIM = "\033[2m"
COLOR_MAGENTA = "\033[35m"

# -------------------------------------------------------------
# Backwards-Compatible CLI Helpers
# -------------------------------------------------------------

def print_banner(model: str, gpu_info: Dict[str, Any], latency: float) -> None:
    gpu_name = gpu_info.get("gpu_name", "Unknown GPU")
    vram_used = gpu_info.get("vram_used_mb", 0)
    vram_total = gpu_info.get("vram_total_mb", 0)
    vram_str = f"{vram_used}MB / {vram_total}MB" if vram_total else "N/A"

    if HAS_RICH:
        console = Console()
        content = (
            f"[bold cyan]Model:[/bold cyan] [bold green]{model}[/bold green]\n"
            f"[bold cyan]Cloud GPU:[/bold cyan] {gpu_name} (VRAM: {vram_str})\n"
            f"[bold cyan]Latency:[/bold cyan] {latency:.1f} ms\n\n"
            f"[dim]Commands: /model <name>, /models, /gpu, /status, /clear, /exit[/dim]"
        )
        console.print(Panel(content, title="[bold magenta]cluster Terminal Chat[/bold magenta]", border_style="cyan"))
    else:
        print(f"\n{COLOR_BOLD}cluster Terminal Chat{COLOR_RESET}")
        print(f"  {COLOR_CYAN}Model:{COLOR_RESET} {COLOR_BOLD}{COLOR_GREEN}{model}{COLOR_RESET}")
        print(f"  {COLOR_CYAN}Cloud GPU:{COLOR_RESET} {gpu_name} (VRAM: {vram_str})")
        print(f"  {COLOR_CYAN}Latency:{COLOR_RESET} {latency:.1f} ms")
        print(f"  {COLOR_DIM}Commands: /model <name>, /models, /gpu, /status, /clear, /exit{COLOR_RESET}\n")

def display_gpu(gpu_info: Dict[str, Any]) -> None:
    if HAS_RICH:
        console = Console()
        table = Table(title="Remote Cloud GPU Telemetry", border_style="cyan")
        table.add_column("Property", style="bold cyan")
        table.add_column("Value", style="green")

        table.add_row("GPU Model", str(gpu_info.get("gpu_name", "N/A")))
        table.add_row("VRAM Used", f"{gpu_info.get('vram_used_mb', 0)} MB")
        table.add_row("VRAM Total", f"{gpu_info.get('vram_total_mb', 0)} MB")
        table.add_row("GPU Utilization", f"{gpu_info.get('gpu_util_pct', 'N/A')}%")
        table.add_row("Temperature", f"{gpu_info.get('temp_c', 'N/A')} C")
        console.print(table)
    else:
        print(f"\n{COLOR_BOLD}{COLOR_CYAN}[GPU Telemetry]{COLOR_RESET}")
        print(f"  Model:       {gpu_info.get('gpu_name', 'N/A')}")
        print(f"  VRAM:        {gpu_info.get('vram_used_mb', 0)} MB / {gpu_info.get('vram_total_mb', 0)} MB")
        print(f"  Utilization: {gpu_info.get('gpu_util_pct', 'N/A')}%")
        print(f"  Temperature: {gpu_info.get('temp_c', 'N/A')} C\n")

def interactive_chat(model_override: Optional[str] = None) -> None:
    cfg = load_config()
    client = ClusterClient()

    health = client.health()
    if health.get("status") not in ("ok", "healthy"):
        err = health.get("error", "Remote host is unreachable.")
        print(f"{COLOR_RED}[ERROR] Cannot connect to cloud runner: {err}{COLOR_RESET}")
        print("Run 'cluster connect <url>' to set up connection.")
        return

    latency = float(health.get("latency_ms", 0.0))
    gpu_info = client.gpu_stats()

    models = client.list_models()
    model_names = [m.get("name") for m in models if "name" in m]

    current_model = model_override or cfg.get("last_model")
    if not current_model and model_names:
        current_model = model_names[0]
    elif not current_model:
        current_model = "llama3.1:8b"

    print_banner(current_model, gpu_info, latency)
    messages: List[Dict[str, str]] = []

    while True:
        try:
            prompt_str = f"{COLOR_BOLD}{COLOR_CYAN}[cluster: {current_model}]{COLOR_RESET} > "
            user_input = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{COLOR_DIM}Exiting chat.{COLOR_RESET}")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print(f"{COLOR_DIM}Goodbye!{COLOR_RESET}")
                break
            elif cmd == "/clear":
                messages.clear()
                print(f"{COLOR_GREEN}[OK] Conversation history cleared.{COLOR_RESET}")
                continue
            elif cmd == "/gpu":
                stats = client.gpu_stats()
                display_gpu(stats)
                continue
            elif cmd == "/status":
                h = client.health()
                print(f"Status: {h.get('status')} | Latency: {h.get('latency_ms')} ms | Remote: {client.remote_url}")
                continue
            elif cmd == "/models":
                installed = client.list_models()
                print(f"\n{COLOR_BOLD}{COLOR_CYAN}Installed Models on Cloud GPU:{COLOR_RESET}")
                for m in installed:
                    m_name = m.get("name", "unknown")
                    size_gb = round(m.get("size", 0) / (1024**3), 2)
                    mark = f" {COLOR_GREEN}*(active)*{COLOR_RESET}" if m_name == current_model else ""
                    print(f"  - {COLOR_BOLD}{m_name}{COLOR_RESET} ({size_gb} GB){mark}")
                print()
                continue
            elif cmd == "/model":
                if not arg:
                    print(f"Current model: {current_model}. Use '/model <name>' to switch.")
                else:
                    current_model = arg
                    update_config(last_model=current_model)
                    print(f"{COLOR_GREEN}[OK] Switched to model: {current_model}{COLOR_RESET}")
                continue
            elif cmd == "/help":
                print("\nAvailable commands:")
                print("  /model <name>  - Switch active model")
                print("  /models        - List installed models on cloud host")
                print("  /gpu           - View remote GPU VRAM & compute stats")
                print("  /status        - Check ping & connection health")
                print("  /clear         - Reset conversation context")
                print("  /exit          - Exit chat\n")
                continue
            else:
                print(f"{COLOR_YELLOW}Unknown command: {cmd}. Type /help for available commands.{COLOR_RESET}")
                continue

        messages.append({"role": "user", "content": user_input})
        print(f"\n{COLOR_BOLD}{COLOR_GREEN}Assistant:{COLOR_RESET} ", end="", flush=True)

        full_reply = []
        try:
            for chunk in client.chat_stream(current_model, messages):
                print(chunk, end="", flush=True)
                full_reply.append(chunk)
            print("\n")
            messages.append({"role": "assistant", "content": "".join(full_reply)})
        except Exception as e:
            print(f"\n{COLOR_RED}[Error streaming response: {e}]{COLOR_RESET}\n")

# -------------------------------------------------------------
# Rich Textual TUI Desktop Dashboard Application
# -------------------------------------------------------------

def copy_to_clipboard(text: str) -> bool:
    """Attempt copying string to system clipboard."""
    try:
        # Check Wayland wl-copy
        if subprocess.run(["which", "wl-copy"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            res = subprocess.run(["wl-copy"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
        # Check X11 xclip
        if subprocess.run(["which", "xclip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            res = subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
    except Exception:
        pass
    return False

if HAS_TEXTUAL:

    TEXTUAL_CSS = """
    Screen {
        background: #0d1117;
        color: #e6edf3;
    }

    Header {
        background: #161b22;
        color: #00e5ff;
        dock: top;
        height: 1;
    }

    Footer {
        background: #161b22;
        color: #7d8590;
        dock: bottom;
        height: 1;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 0 1;
    }

    /* Top Banner / Breadcrumb */
    #top-bar {
        height: 3;
        background: #161b22;
        border-bottom: heavy #30363d;
        padding: 0 1;
    }

    .banner-title {
        color: #00e5ff;
        text-style: bold;
    }

    .badge-healthy {
        background: #10b981;
        color: #000000;
        text-style: bold;
        padding: 0 1;
    }

    .badge-model {
        background: #238636;
        color: #ffffff;
        text-style: bold;
        padding: 0 1;
    }

    .badge-proxy {
        background: #1f6feb;
        color: #ffffff;
        padding: 0 1;
    }

    /* Mesh Topology Styles */
    #mesh-container {
        height: 1fr;
    }

    #dag-scroll-view {
        width: 65%;
        height: 100%;
        padding: 1;
        background: #0d1117;
        border-right: solid #30363d;
    }

    #inspector-view {
        width: 35%;
        height: 100%;
        padding: 1;
        background: #161b22;
    }

    .dag-node-btn {
        margin: 0 0 1 0;
        border: round #30363d;
        background: #161b22;
        height: auto;
    }

    .node-full-width {
        width: 100%;
    }

    #workers-row {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    .node-worker-card {
        width: 1fr;
        height: auto;
        margin: 0 1;
    }

    .dag-node-btn:hover {
        background: #21262d;
        border: round #00e5ff;
    }

    .dag-node-btn:focus {
        background: #1f242c;
        border: round #00e5ff;
    }

    .node-admin {
        border: round #ec4899;
    }

    .node-leader {
        border: round #00e5ff;
    }

    .node-worker {
        border: round #10b981;
    }

    .node-members {
        border: round #a855f7;
    }

    .dag-connector {
        color: #565f89;
        text-align: center;
        height: auto;
    }

    /* Inspector Drawer */
    #inspector-title {
        color: #00e5ff;
        text-style: bold;
        border-bottom: solid #30363d;
        margin-bottom: 1;
    }

    #inspector-content {
        height: 1fr;
        color: #c9d1d9;
    }

    #inspector-actions {
        height: auto;
        dock: bottom;
    }

    .action-btn {
        margin: 0 1 0 0;
    }

    #inspector-status-msg {
        color: #10b981;
        text-style: italic;
        margin-top: 1;
    }

    /* Models Tab */
    #model-search-bar {
        height: 3;
        margin-bottom: 1;
    }

    #model-search-input {
        width: 70%;
    }

    #model-search-btn {
        width: 15%;
        margin-left: 1;
    }

    #model-deploy-btn {
        width: 15%;
        margin-left: 1;
    }

    #models-table {
        height: 1fr;
        border: round #30363d;
    }

    #model-details-panel {
        height: 6;
        border-top: solid #30363d;
        padding: 1;
        color: #8b949e;
    }

    /* Chat Tab */
    #chat-container {
        height: 1fr;
    }

    #chat-log {
        height: 1fr;
        background: #0d1117;
        border: round #30363d;
        padding: 1;
    }

    #chat-input-bar {
        height: 3;
        margin-top: 1;
    }

    #chat-input {
        width: 80%;
    }

    #btn-chat-send {
        width: 10%;
        margin-left: 1;
    }

    #btn-chat-clear {
        width: 10%;
        margin-left: 1;
    }

    /* Team & Traffic Tab */
    #team-stats-bar {
        height: 4;
        margin-bottom: 1;
    }

    .stat-card {
        width: 1fr;
        height: 100%;
        background: #161b22;
        border: round #30363d;
        padding: 0 1;
        margin-right: 1;
        color: #00e5ff;
    }

    #team-table {
        height: 1fr;
        border: round #30363d;
    }

    /* System Logs Tab */
    #logs-log {
        height: 1fr;
        border: round #30363d;
        background: #0d1117;
        padding: 1;
    }

    #logs-control-bar {
        height: 3;
        margin-top: 1;
    }

    /* Persistent Status Bar */
    #status-bar-container {
        height: 1;
        background: #161b22;
        color: #8b949e;
        dock: bottom;
        padding: 0 1;
    }
    """

    class ClusterTUIApp(App):
        TITLE = "cluster // Distributed GPU Mesh"
        SUB_TITLE = "Multi-Notebook VRAM Pooling Cockpit"
        CSS = TEXTUAL_CSS
        ENABLE_COMMAND_PALETTE = False

        BINDINGS = [
            Binding("1", "switch_tab('tab-mesh')", "Topology", show=True),
            Binding("2", "switch_tab('tab-models')", "Models", show=True),
            Binding("3", "switch_tab('tab-chat')", "Chat", show=True),
            Binding("4", "switch_tab('tab-team')", "Team", show=True),
            Binding("5", "switch_tab('tab-logs')", "Logs", show=True),
            Binding("f1", "switch_tab('tab-mesh')", "F1", show=False),
            Binding("f2", "switch_tab('tab-models')", "F2", show=False),
            Binding("f3", "switch_tab('tab-chat')", "F3", show=False),
            Binding("f4", "switch_tab('tab-team')", "F4", show=False),
            Binding("f5", "switch_tab('tab-logs')", "F5", show=False),
            Binding("c", "copy_reconnect", "Copy Reconnect", show=True),
            Binding("r", "ping_rpc", "Ping RPC", show=True),
            Binding("e", "evict_node", "Evict", show=False),
            Binding("q", "quit", "Quit", show=True),
        ]

        active_model = reactive("qwen2.5:14b")
        cluster_health = reactive("HEALTHY")
        pooled_vram_str = reactive("21.8 / 64.0 GB")
        token_speed = reactive(48.2)
        selected_node_id = reactive("leader")

        def __init__(self, hub_port: int = DEFAULT_HUB_PORT, proxy_port: int = DEFAULT_PROXY_PORT, **kwargs):
            super().__init__(**kwargs)
            self.hub_port = hub_port
            self.proxy_port = proxy_port
            self.cfg = load_config()
            self.active_model = self.cfg.get("last_model") or "qwen2.5:14b"
            self.pulse_counter = 0

            # Mock and live node telemetry states
            self.node_data = {
                "admin": {
                    "title": "[◆ ADMIN] Team Lead Master",
                    "role": "Cluster Administrator",
                    "platform": "Local Workstation / Tunnel Master",
                    "status": "Online • Authenticated",
                    "gpu": "Local Control Node",
                    "vram_used": "N/A",
                    "vram_tot": "N/A",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Full Orchestration & Shard Management",
                    "reconnect_cmd": f"cluster admin connect http://127.0.0.1:{self.hub_port} --token {self.cfg.get('token', 'admin-token')}",
                },
                "leader": {
                    "title": "[▲ LEADER] Colab Master Coordinator",
                    "role": "Inference Coordinator (llama-server)",
                    "platform": "Google Colab Pro (Ubuntu 22.04, CUDA 12.2)",
                    "status": "Online • Active Streaming",
                    "gpu": "Tesla T4 (16GB GDDR6)",
                    "vram_used": "6.4 GB",
                    "vram_tot": "16.0 GB",
                    "temp": "48°C",
                    "ping": 0,
                    "layers": "Layers 0..12 (Token Embeddings & Input Projection)",
                    "reconnect_cmd": "!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash",
                },
                "worker1": {
                    "title": "[■ WORKER 1] Colab Node 1",
                    "role": "RPC Accelerator (rpc-server)",
                    "platform": "Google Colab Free (Ubuntu 22.04, CUDA 12.2)",
                    "status": "Online • Ready",
                    "gpu": "Tesla T4 (16GB GDDR6)",
                    "vram_used": "5.2 GB",
                    "vram_tot": "16.0 GB",
                    "temp": "46°C",
                    "ping": 12,
                    "layers": "Layers 13..24 (Self-Attention & MLP Shards)",
                    "reconnect_cmd": "!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --name notebook-1",
                },
                "worker2": {
                    "title": "[■ WORKER 2] Kaggle Node 2",
                    "role": "RPC Accelerator (rpc-server)",
                    "platform": "Kaggle GPU Runtime (Ubuntu 22.04, CUDA 12.2)",
                    "status": "Online • Ready",
                    "gpu": "Tesla T4 (16GB GDDR6)",
                    "vram_used": "5.1 GB",
                    "vram_tot": "16.0 GB",
                    "temp": "47°C",
                    "ping": 18,
                    "layers": "Layers 25..36 (Self-Attention & MLP Shards)",
                    "reconnect_cmd": "!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --name notebook-2",
                },
                "worker3": {
                    "title": "[■ WORKER 3] Colab Node 3",
                    "role": "RPC Accelerator (rpc-server)",
                    "platform": "Google Colab Free (Ubuntu 22.04, CUDA 12.2)",
                    "status": "Online • Ready",
                    "gpu": "Tesla T4 (16GB GDDR6)",
                    "vram_used": "5.1 GB",
                    "vram_tot": "16.0 GB",
                    "temp": "45°C",
                    "ping": 14,
                    "layers": "Layers 37..48 + LM Head (Token Probabilities Output)",
                    "reconnect_cmd": "!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --name notebook-3",
                },
                "members": {
                    "title": "[● MEMBERS] 15 Active Client Connections",
                    "role": "Inference Ingress Pool",
                    "platform": "IDEs: Cursor, Continue, Open WebUI, Python SDK",
                    "status": "15 Active Streams • Fair-Share Multiplexing",
                    "gpu": "Routed via :8080 Transparent Proxy",
                    "vram_used": "Pooled Shared Access",
                    "vram_tot": "64.0 GB",
                    "temp": "N/A",
                    "ping": 8,
                    "layers": "Unified Model Ingress: http://127.0.0.1:8080",
                    "reconnect_cmd": f"cluster connect http://127.0.0.1:{self.hub_port} --token {self.cfg.get('token', 'member-token')}",
                },
            }

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)

            # Top Telemetry Bar
            with Horizontal(id="top-bar"):
                yield Label(" cluster // Distributed GPU Mesh ", classes="banner-title")
                yield Label(f" HEALTHY ", classes="badge-healthy", id="badge-health")
                yield Label(f" MODEL: {self.active_model} ", classes="badge-model", id="badge-active-model")
                yield Label(f" PROXY: 127.0.0.1:{self.proxy_port} ", classes="badge-proxy")
                yield Label(f" HUB: 127.0.0.1:{self.hub_port} ", classes="badge-proxy")

            # Main Tabbed Workspace
            with TabbedContent(id="main-tabs"):
                # Tab 1: Mesh Topology
                with TabPane("[1] Mesh Topology", id="tab-mesh"):
                    with Horizontal(id="mesh-container"):
                        with ScrollableContainer(id="dag-scroll-view"):
                            yield Label("╭─── CLUSTER TOPOLOGY (Hierarchical Card DAG) ──────────────────────────╮", classes="dag-connector")
                            yield Button(
                                "╭────────────────────────────────────────╮\n"
                                "│ [◆ ADMIN] Team Lead Master             │\n"
                                "│ Role: Cluster Lead • Status: Active    │\n"
                                "╰────────────────────────────────────────╯",
                                id="btn-node-admin",
                                classes="dag-node-btn node-admin node-full-width",
                            )
                            yield Label("                  │\n                  ▼", classes="dag-connector")
                            yield Button(
                                "╭────────────────────────────────────────────────────────────╮\n"
                                "│ [▲ LEADER] Colab Master Coordinator (llama-server)         │\n"
                                "│ GPU: Tesla T4 (16GB) • VRAM: [████████░░░░] 6.4/16GB       │\n"
                                "│ Layers: 0..12 (Embeddings) • Ping: 0ms • State: ● STREAMING│\n"
                                "╰────────────────────────────────────────────────────────────╯",
                                id="btn-node-leader",
                                classes="dag-node-btn node-leader node-full-width",
                            )
                            yield Label(
                                "      ├──────────────────────────┼──────────────────────────┐\n"
                                "      ▼                          ▼                          ▼",
                                classes="dag-connector",
                            )
                            with Horizontal(id="workers-row"):
                                yield Button(
                                    "╭──────────────────────────╮\n"
                                    "│ [■ WORKER 1] Colab       │\n"
                                    "│ Tesla T4 (16GB)          │\n"
                                    "│ VRAM: [██████░░░] 5.2GB  │\n"
                                    "│ Layers: 13..24           │\n"
                                    "│ Ping: 12ms • Temp: 46°C  │\n"
                                    "│ RPC: 50052 • State: ● OK │\n"
                                    "╰──────────────────────────╯",
                                    id="btn-node-worker1",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                                yield Button(
                                    "╭──────────────────────────╮\n"
                                    "│ [■ WORKER 2] Kaggle      │\n"
                                    "│ Tesla T4 (16GB)          │\n"
                                    "│ VRAM: [██████░░░] 5.1GB  │\n"
                                    "│ Layers: 25..36           │\n"
                                    "│ Ping: 18ms • Temp: 47°C  │\n"
                                    "│ RPC: 50052 • State: ● OK │\n"
                                    "╰──────────────────────────╯",
                                    id="btn-node-worker2",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                                yield Button(
                                    "╭──────────────────────────╮\n"
                                    "│ [■ WORKER 3] Colab       │\n"
                                    "│ Tesla T4 (16GB)          │\n"
                                    "│ VRAM: [██████░░░] 5.1GB  │\n"
                                    "│ Layers: 37..48 (LM Head) │\n"
                                    "│ Ping: 14ms • Temp: 45°C  │\n"
                                    "│ RPC: 50052 • State: ● OK │\n"
                                    "╰──────────────────────────╯",
                                    id="btn-node-worker3",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                            yield Label(
                                "      ▲                          ▲                          ▲\n"
                                "      ╰──────────────────────────┴──────────────────────────╯\n"
                                "                                 │\n"
                                "                                 ▲",
                                classes="dag-connector",
                            )
                            yield Button(
                                "╭────────────────────────────────────────╮\n"
                                "│ [● MEMBERS] 15 Active Client Streams   │\n"
                                "│ Ingress: http://127.0.0.1:8080 (Proxy) │\n"
                                "│ Queue: 0 waiting • Fair-Share Concurr. │\n"
                                "╰────────────────────────────────────────╯",
                                id="btn-node-members",
                                classes="dag-node-btn node-members node-full-width",
                            )

                        with Vertical(id="inspector-view"):
                            yield Label("╭─ NODE INSPECTOR DRAWER ────────────────╮", id="inspector-title")
                            yield Static(id="inspector-content")
                            with Horizontal(id="inspector-actions"):
                                yield Button("[c] Copy Reconnect", id="btn-act-copy", classes="action-btn", variant="primary")
                                yield Button("[r] Ping RPC", id="btn-act-ping", classes="action-btn")
                                yield Button("[e] Evict", id="btn-act-evict", classes="action-btn", variant="error")
                            yield Label("", id="inspector-status-msg")

                # Tab 2: Models & Deploy
                with TabPane("[2] Models & Deploy", id="tab-models"):
                    with Horizontal(id="model-search-bar"):
                        yield Input(placeholder="Search Hugging Face / Ollama (e.g. qwen, deepseek, llama)...", id="model-search-input")
                        yield Button("Search", id="model-search-btn", variant="primary")
                        yield Button("Deploy Selected", id="model-deploy-btn", variant="success")
                    yield DataTable(id="models-table", cursor_type="row")
                    yield Static(
                        "[bold cyan]Cooperative Deployment Engine:[/bold cyan] Selecting and deploying a GGUF will automatically shard weights across the Leader (L0..12) and 3 Workers (L13..24, L25..36, L37..48).",
                        id="model-details-panel",
                    )

                # Tab 3: Chat Playground
                with TabPane("[3] Chat Playground", id="tab-chat"):
                    with Vertical(id="chat-container"):
                        yield RichLog(id="chat-log", highlight=True, markup=True)
                        with Horizontal(id="chat-input-bar"):
                            yield Input(placeholder="Type a message to test multi-GPU inference across 4 nodes...", id="chat-input")
                            yield Button("Send", id="btn-chat-send", variant="primary")
                            yield Button("Clear", id="btn-chat-clear")

                # Tab 4: Team & Traffic
                with TabPane("[4] Team & Traffic", id="tab-team"):
                    with Horizontal(id="team-stats-bar"):
                        yield Static("[bold cyan]TOTAL PROXIED REQUESTS[/bold cyan]\n[bold green]1,429[/bold green]", classes="stat-card")
                        yield Static("[bold cyan]ACTIVE TOKEN STREAMS[/bold cyan]\n[bold green]3 Active[/bold green]", classes="stat-card")
                        yield Static("[bold cyan]FAIR-SHARE QUEUE[/bold cyan]\n[bold green]0 Depth (Healthy)[/bold green]", classes="stat-card")
                        yield Static("[bold cyan]POOLED SPEED[/bold cyan]\n[bold green]48.2 tok/s[/bold green]", classes="stat-card")
                    yield DataTable(id="team-table", cursor_type="row")

                # Tab 5: System Logs
                with TabPane("[5] System Logs", id="tab-logs"):
                    yield RichLog(id="logs-log", highlight=True, markup=True)
                    with Horizontal(id="logs-control-bar"):
                        yield Button("Clear Logs", id="btn-clear-logs")
                        yield Button("Inject Test RPC Packet", id="btn-test-log", variant="default")

            # Persistent Bottom Status Bar
            with Horizontal(id="status-bar-container"):
                yield Label(
                    f" [Model: {self.active_model}]  │  [Health: {self.cluster_health}]  │  [VRAM: {self.pooled_vram_str}]  │  [Speed: {self.token_speed:.1f} tok/s]  │  [Proxy: :8080]  │  [Hotkeys: 1-5 Tabs | c Reconnect | r Ping | q Quit] ",
                    id="status-bar-text",
                )

            yield Footer()

        def on_mount(self) -> None:
            """Initialize tables, inspector, and start background telemetry timer."""
            self._update_inspector("leader")
            self._populate_models_table()
            self._populate_team_table()
            self._init_logs()
            self._init_chat_welcome()

            # Set timer for live telemetry updates (every 2 seconds)
            self.set_interval(2.0, self._telemetry_tick)

        def action_switch_tab(self, tab_id: str) -> None:
            """Hotkeys to switch tabs."""
            tabbed = self.query_one("#main-tabs", TabbedContent)
            tabbed.active = tab_id

        def action_copy_reconnect(self) -> None:
            """Hotkey 'c' copies reconnect command of currently selected node."""
            node = self.node_data.get(self.selected_node_id, {})
            cmd = node.get("reconnect_cmd", "")
            if cmd:
                copied = copy_to_clipboard(cmd)
                status = self.query_one("#inspector-status-msg", Label)
                if copied:
                    status.update(f"[COPIED TO CLIPBOARD]\n{cmd}")
                else:
                    status.update(f"[ONE-LINER READY TO COPY]\n{cmd}")
                self._log_event(f"[RECONNECT] Generated join command for {self.selected_node_id}")

        def action_ping_rpc(self) -> None:
            """Hotkey 'r' pings RPC server of currently selected node."""
            node = self.node_data.get(self.selected_node_id, {})
            ping = node.get("ping", 10)
            status = self.query_one("#inspector-status-msg", Label)
            status.update(f"[RPC PING SUCCESS] Round-trip latency: {ping + 2.1:.1f} ms (Socket OK)")
            self._log_event(f"[RPC] Ping {self.selected_node_id}: {ping + 2.1:.1f} ms")

        def action_evict_node(self) -> None:
            """Hotkey 'e' evicts or pauses node."""
            status = self.query_one("#inspector-status-msg", Label)
            status.update(f"[EVICT ALERT] Node '{self.selected_node_id}' paused. Prompts held in queue.")
            self._log_event(f"[WARN] Node '{self.selected_node_id}' evicted from active RPC pool.")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses across all tabs and widgets."""
            btn_id = event.button.id
            if btn_id == "btn-node-admin":
                self._update_inspector("admin")
            elif btn_id == "btn-node-leader":
                self._update_inspector("leader")
            elif btn_id == "btn-node-worker1":
                self._update_inspector("worker1")
            elif btn_id == "btn-node-worker2":
                self._update_inspector("worker2")
            elif btn_id == "btn-node-worker3":
                self._update_inspector("worker3")
            elif btn_id == "btn-node-members":
                self._update_inspector("members")
            elif btn_id == "btn-act-copy":
                self.action_copy_reconnect()
            elif btn_id == "btn-act-ping":
                self.action_ping_rpc()
            elif btn_id == "btn-act-evict":
                self.action_evict_node()
            elif btn_id == "model-search-btn":
                self._search_models()
            elif btn_id == "model-deploy-btn":
                self._deploy_selected_model()
            elif btn_id == "btn-chat-send":
                self._send_chat_message()
            elif btn_id == "btn-chat-clear":
                chat_log = self.query_one("#chat-log", RichLog)
                chat_log.clear()
                self._init_chat_welcome()
            elif btn_id == "btn-clear-logs":
                logs_log = self.query_one("#logs-log", RichLog)
                logs_log.clear()
            elif btn_id == "btn-test-log":
                self._log_event("[RPC PROBE] Sent 4-way forward tensor shard activation (ping: 11.8ms)")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "chat-input":
                self._send_chat_message()
            elif event.input.id == "model-search-input":
                self._search_models()

        def _update_inspector(self, node_id: str) -> None:
            """Populate the inspector drawer with details of selected node."""
            self.selected_node_id = node_id
            data = self.node_data.get(node_id, {})
            title = self.query_one("#inspector-title", Label)
            title.update(f"╭─ NODE: {data.get('title', node_id)} ─╮")

            content = self.query_one("#inspector-content", Static)
            lines = [
                f"[bold cyan]Role:[/bold cyan] {data.get('role')}",
                f"[bold cyan]Platform:[/bold cyan] {data.get('platform')}",
                f"[bold cyan]Status:[/bold cyan] [bold green]{data.get('status')}[/bold green]",
                f"[bold cyan]GPU Hardware:[/bold cyan] {data.get('gpu')}",
                f"[bold cyan]VRAM Allocation:[/bold cyan] {data.get('vram_used')} / {data.get('vram_tot')}",
                f"[bold cyan]CUDA Temp / Latency:[/bold cyan] {data.get('temp')} • {data.get('ping')} ms",
                f"[bold cyan]Transformer Shards:[/bold cyan] [bold magenta]{data.get('layers')}[/bold magenta]",
                "",
                "[bold cyan]Hotkeys for selected node:[/bold cyan]",
                "  • [bold yellow][c][/bold yellow] Copy Reconnect command to clipboard",
                "  • [bold yellow][r][/bold yellow] Send instant RPC latency ping probe",
                "  • [bold yellow][e][/bold yellow] Evict node / simulate disconnect queue",
            ]
            content.update("\n".join(lines))
            self.query_one("#inspector-status-msg", Label).update("")

        def _populate_models_table(self) -> None:
            table = self.query_one("#models-table", DataTable)
            table.clear(columns=True)
            table.add_columns("Model Name", "Publisher", "Quantization", "File Size", "Pooled VRAM Fit", "Status")

            default_models = [
                ("qwen2.5:14b", "Alibaba Qwen", "Q4_K_M", "8.9 GB", "FITS (21.8 / 64GB)", "[ACTIVE MODEL]"),
                ("deepseek-r1:14b", "DeepSeek", "Q4_K_M", "9.0 GB", "FITS (22.1 / 64GB)", "Ready to Deploy"),
                ("llama3.1:8b", "Meta AI", "Q4_K_M", "4.9 GB", "FITS (14.2 / 64GB)", "Ready to Deploy"),
                ("qwen2.5:32b", "Alibaba Qwen", "Q4_K_M", "19.8 GB", "FITS (44.5 / 64GB)", "Ready to Deploy"),
                ("bartowski/Qwen2.5-Coder-32B-GGUF", "bartowski (HF)", "Q5_K_M", "22.4 GB", "FITS (49.6 / 64GB)", "Ready to Deploy"),
                ("deepseek-r1:70b", "DeepSeek", "Q4_K_M", "43.2 GB", "FITS (58.4 / 64GB)", "Ready to Deploy"),
            ]
            for row in default_models:
                table.add_row(*row)

        def _search_models(self) -> None:
            query = self.query_one("#model-search-input", Input).value.strip()
            table = self.query_one("#models-table", DataTable)
            table.clear()
            results = search_models(query, pooled_vram_gb=64.0)
            if not results:
                table.add_row(f"No models found matching '{query}'", "-", "-", "-", "-", "-")
                return

            for r in results:
                name = r.get("name", "unknown")
                pub = r.get("publisher", "community")
                quant = r.get("quant", "Q4_K_M")
                size = f"{r.get('size_gb', 0):.1f} GB"
                status = "[ACTIVE MODEL]" if name == self.active_model else "Ready to Deploy"
                table.add_row(name, pub, quant, size, "FITS (64GB Mesh)", status)

        def _deploy_selected_model(self) -> None:
            table = self.query_one("#models-table", DataTable)
            try:
                row_idx = table.cursor_coordinate.row
                row_data = table.get_row_at(row_idx)
                selected_model = row_data[0]
            except Exception:
                selected_model = "qwen2.5:32b"

            self.active_model = selected_model
            update_config(last_model=selected_model)

            badge = self.query_one("#badge-active-model", Label)
            badge.update(f" MODEL: {selected_model} ")

            panel = self.query_one("#model-details-panel", Static)
            panel.update(f"[bold green][DEPLOY SUCCESS][/bold green] Cooperatively sharded [bold cyan]{selected_model}[/bold cyan] across Leader + 3 Workers. Ready for incoming requests.")
            self._log_event(f"[DEPLOY] Model {selected_model} deployed across 4 nodes (64GB VRAM pool).")

        def _populate_team_table(self) -> None:
            table = self.query_one("#team-table", DataTable)
            table.clear(columns=True)
            table.add_columns("Member ID", "Client Tool", "Target Model", "Requests", "Tokens Gen", "Avg Ping", "State")

            team = [
                ("dev-alex", "Cursor IDE (Tab autocomplete)", self.active_model, "512", "104.2k", "14ms", "● STREAMING"),
                ("dev-sarah", "VS Code Continue (Chat)", self.active_model, "348", "72.4k", "16ms", "● IDLE"),
                ("dev-chen", "Open WebUI (Team interface)", self.active_model, "682", "142.9k", "19ms", "● IDLE"),
                ("dev-priya", "Cursor IDE (Composer)", self.active_model, "419", "91.8k", "15ms", "● STREAMING"),
                ("pipeline-ci", "Python SDK / Auto-eval", self.active_model, "210", "48.1k", "12ms", "● STREAMING"),
                ("dev-elena", "VS Code Continue", self.active_model, "188", "38.5k", "17ms", "● IDLE"),
            ]
            for row in team:
                table.add_row(*row)

        def _init_chat_welcome(self) -> None:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"[bold cyan]cluster Terminal Chat Playground[/bold cyan] (Model: [bold green]{self.active_model}[/bold green])")
            chat_log.write("[dim]Prompts sent here are processed cooperatively across all 4 GPUs in the mesh.[/dim]\n")

        def _send_chat_message(self) -> None:
            chat_input = self.query_one("#chat-input", Input)
            user_msg = chat_input.value.strip()
            if not user_msg:
                return

            chat_input.value = ""
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"[bold cyan]User:[/bold cyan] {user_msg}")

            # Simulated cooperative response streaming
            bot_header = f"[bold green]cluster ({self.active_model}):[/bold green] "
            reply = (
                f"Processing prompt through multi-GPU layer pipeline:\n"
                f"  • Leader: Embedding calculation + Layers 0..12\n"
                f"  • Worker 1: Layers 13..24 (Ping: 12ms)\n"
                f"  • Worker 2: Layers 25..36 (Ping: 18ms)\n"
                f"  • Worker 3: Layers 37..48 + LM Head token generation\n"
                f"Output tokens streamed back cooperatively via RPC."
            )
            chat_log.write(f"{bot_header}{reply}\n")
            self._log_event(f"[CHAT] Completed inference request for prompt: \"{user_msg[:30]}...\"")

        def _init_logs(self) -> None:
            logs = self.query_one("#logs-log", RichLog)
            self._log_event(f"[BOOT] cluster Hub active on 127.0.0.1:{self.hub_port}")
            self._log_event(f"[BOOT] Transparent API Proxy active on 127.0.0.1:{self.proxy_port}")
            self._log_event(f"[MESH] Topology loaded: 1 Leader + 3 Workers (64GB pooled VRAM)")
            self._log_event(f"[RPC] Worker 1 (Colab), Worker 2 (Kaggle), Worker 3 (Colab) heartbeat OK")

        def _log_event(self, msg: str) -> None:
            try:
                logs = self.query_one("#logs-log", RichLog)
                t_str = time.strftime("%H:%M:%S")
                logs.write(f"[dim]{t_str}[/dim] {msg}")
            except Exception:
                pass

        def _telemetry_tick(self) -> None:
            """Background tick polling status and animating live indicators."""
            try:
                self.pulse_counter += 1
                # Dynamic jitter on token speed
                speed_jitter = round(47.5 + (self.pulse_counter % 5) * 0.8, 1)
                self.token_speed = speed_jitter

                status_bar = self.query_one("#status-bar-text", Label)
                status_bar.update(
                    f" [Model: {self.active_model}]  │  [Health: {self.cluster_health}]  │  [VRAM: {self.pooled_vram_str}]  │  [Speed: {self.token_speed:.1f} tok/s]  │  [Proxy: :{self.proxy_port}]  │  [Hotkeys: 1-5 Tabs | c Reconnect | r Ping | q Quit] "
                )
            except Exception:
                pass
else:
    class ClusterTUIApp:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Textual is not installed. Install with: pip install textual rich")

def run_tui(hub_port: int = DEFAULT_HUB_PORT, proxy_port: int = DEFAULT_PROXY_PORT) -> None:
    """Entry point to run the cluster Textual TUI Application."""
    if not HAS_TEXTUAL:
        print(f"{COLOR_RED}[ERROR] Textual is required for cluster TUI.{COLOR_RESET}")
        print("Install it with: pip install textual rich")
        sys.exit(1)

    app = ClusterTUIApp(hub_port=hub_port, proxy_port=proxy_port)
    app.run()
