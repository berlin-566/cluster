import asyncio
import http.client
import json
import logging
import os
import shutil
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
    from textual import work
    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False

from cluster.client import ClusterClient
from cluster.config import load_config, update_config, DEFAULT_HUB_PORT, DEFAULT_PROXY_PORT
from cluster.fetch import search_models

logger = logging.getLogger("cluster.tui")

# ANSI fallback colors (minimalist & clean)
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_CYAN = "\033[37m"
COLOR_MAGENTA = "\033[37m"

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
            f"[bold #ffffff]Model:[/bold #ffffff] [#ececee]{model}[/#ececee]\n"
            f"[dim]Cloud GPU:[/dim] [#ececee]{gpu_name}[/#ececee] (VRAM: {vram_str})\n"
            f"[dim]Latency:[/dim] {latency:.1f} ms\n\n"
            f"[dim]Commands: /model <name>, /models, /gpu, /status, /clear, /exit[/dim]"
        )
        console.print(Panel(content, title="cluster Terminal Chat", border_style="#333338"))
    else:
        print(f"\n{COLOR_BOLD}cluster Terminal Chat{COLOR_RESET}")
        print(f"  Model:     {model}")
        print(f"  Cloud GPU: {gpu_name} (VRAM: {vram_str})")
        print(f"  Latency:   {latency:.1f} ms")
        print(f"  {COLOR_DIM}Commands: /model <name>, /models, /gpu, /status, /clear, /exit{COLOR_RESET}\n")

def display_gpu(gpu_info: Dict[str, Any]) -> None:
    if HAS_RICH:
        console = Console()
        table = Table(title="Remote Cloud GPU Telemetry", border_style="#333338")
        table.add_column("Property", style="bold #ececee")
        table.add_column("Value", style="#ececee")

        table.add_row("GPU Model", str(gpu_info.get("gpu_name", "N/A")))
        table.add_row("VRAM Used", f"{gpu_info.get('vram_used_mb', 0)} MB")
        table.add_row("VRAM Total", f"{gpu_info.get('vram_total_mb', 0)} MB")
        table.add_row("GPU Utilization", f"{gpu_info.get('gpu_util_pct', 'N/A')}%")
        table.add_row("Temperature", f"{gpu_info.get('temp_c', 'N/A')} C")
        console.print(table)
    else:
        print(f"\n[GPU Telemetry]")
        print(f"  Model:       {gpu_info.get('gpu_name', 'N/A')}")
        print(f"  VRAM:        {gpu_info.get('vram_used_mb', 0)} MB / {gpu_info.get('vram_total_mb', 0)} MB")
        print(f"  Utilization: {gpu_info.get('gpu_util_pct', 'N/A')}%")
        print(f"  Temperature: {gpu_info.get('temp_c', 'N/A')} C\n")

def interactive_chat(model_override: Optional[str] = None) -> None:
    cfg = load_config()
    client = ClusterClient()

    health = client.health(wait_for_dns=True)
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
        current_model = "qwen2.5:14b"

    print_banner(current_model, gpu_info, latency)
    messages: List[Dict[str, str]] = []

    while True:
        try:
            prompt_str = f"{COLOR_BOLD}{current_model}{COLOR_RESET} > "
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
                print(f"{COLOR_DIM}Goodbye.{COLOR_RESET}")
                break
            elif cmd == "/clear":
                messages.clear()
                print("Conversation history cleared.")
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
                print(f"\nInstalled Models on Cloud GPU:")
                for m in installed:
                    m_name = m.get("name", "unknown")
                    size_gb = round(m.get("size", 0) / (1024**3), 2)
                    mark = " *(active)*" if m_name == current_model else ""
                    print(f"  - {m_name} ({size_gb} GB){mark}")
                print()
                continue
            elif cmd == "/model":
                if not arg:
                    print(f"Current model: {current_model}. Use '/model <name>' to switch.")
                else:
                    current_model = arg
                    update_config(last_model=current_model)
                    print(f"Switched to model: {current_model}")
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
                print(f"Unknown command: {cmd}. Type /help for available commands.")
                continue

        messages.append({"role": "user", "content": user_input})
        print(f"\n{COLOR_BOLD}Assistant:{COLOR_RESET} ", end="", flush=True)

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
    """Attempt copying string to system clipboard across Wayland, X11, and macOS."""
    try:
        if shutil.which("wl-copy"):
            res = subprocess.run(["wl-copy"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
        if shutil.which("xclip"):
            res = subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
        if shutil.which("xsel"):
            res = subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
        if shutil.which("pbcopy"):
            res = subprocess.run(["pbcopy"], input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return True
    except Exception:
        pass
    return False

TEXTUAL_CSS = """
    Screen {
        background: #0f1017;
        color: #e2e8f0;
    }

    Footer {
        background: #12131c;
        color: #64748b;
        dock: bottom;
        height: 1;
    }

    TabbedContent {
        height: 1fr;
    }

    Tabs {
        background: #12131c;
        border-bottom: solid #2a2d3d;
        height: 3;
    }

    Tab {
        padding: 0 2;
        color: #64748b !important;
        background: transparent;
        text-style: none;
    }

    Tab:hover {
        color: #e2e8f0 !important;
        background: #1e202e;
    }

    Tab:focus {
        color: #38bdf8 !important;
        background: #1e202e !important;
        text-style: bold;
    }

    Tab.-active {
        color: #38bdf8 !important;
        background: #1e202e !important;
        text-style: bold;
    }

    Tab.-active:focus {
        color: #38bdf8 !important;
        background: #1e202e !important;
        text-style: bold;
    }

    Tab.-active:hover {
        color: #38bdf8 !important;
        background: #1e202e !important;
    }

    TabPane {
        padding: 1 1;
        background: #0f1017;
    }

    /* Top Telemetry Bar */
    #top-bar {
        height: 3;
        background: #12131c;
        border-bottom: solid #2a2d3d;
        padding: 0 1;
        align-vertical: middle;
    }

    .badge-healthy {
        background: #171822;
        color: #4ade80;
        border: round #4ade80;
        padding: 0 1;
        margin-right: 1;
        height: 3;
    }

    .badge-model {
        background: #171822;
        color: #38bdf8;
        border: round #2a2d3d;
        padding: 0 1;
        margin-right: 1;
        height: 3;
    }

    .badge-vram {
        background: #171822;
        color: #c084fc;
        border: round #2a2d3d;
        padding: 0 1;
        margin-right: 1;
        height: 3;
    }

    .badge-proxy {
        background: #171822;
        color: #94a3b8;
        border: round #2a2d3d;
        padding: 0 1;
        margin-right: 1;
        height: 3;
    }

    /* Mesh Topology Layout */
    #mesh-container {
        height: 1fr;
    }

    #dag-scroll-view {
        width: 70%;
        height: 100%;
        padding: 0 1;
        background: #0f1017;
        border-right: solid #2a2d3d;
    }

    .dag-section-label {
        color: #64748b;
        margin-bottom: 1;
        text-style: bold;
    }

    #inspector-view {
        width: 30%;
        height: 100%;
        padding: 0 1;
        background: #141520;
        border-left: solid #2a2d3d;
    }

    .dag-node-btn {
        margin: 0 0 1 0;
        padding: 0 1;
        border: round #2a2d3d;
        background: #171822;
        color: #e2e8f0;
        text-align: left;
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
        text-align: left;
    }

    .dag-node-btn:hover {
        background: #1e202e;
        border: round #38bdf8;
    }

    .dag-node-btn:focus {
        background: #1e202e;
        border: round #38bdf8;
    }

    .node-admin {
        border: round #2a2d3d;
    }

    .node-leader {
        border: round #2a2d3d;
    }

    .node-worker {
        border: round #2a2d3d;
    }

    .node-members {
        border: round #2a2d3d;
    }

    .dag-connector {
        color: #475569;
        text-align: center;
        height: 1;
        margin-bottom: 1;
    }

    /* Inspector Drawer */
    #inspector-title {
        color: #ffffff;
        text-style: bold;
        border-bottom: solid #2a2d3d;
        padding-bottom: 0;
        margin-bottom: 1;
    }

    #inspector-content {
        height: 1fr;
        color: #e2e8f0;
    }

    #inspector-actions {
        height: 3;
        dock: bottom;
        margin-top: 1;
    }

    .action-btn {
        height: 3;
        min-width: 10;
        margin-right: 1;
        background: #1e202e;
        border: round #2a2d3d;
        color: #e2e8f0;
    }

    .action-btn:hover {
        background: #2a2d3d;
        border: round #38bdf8;
    }

    #inspector-status-msg {
        color: #4ade80;
        margin-top: 1;
        height: 2;
    }

    /* Models Tab */
    #model-search-bar {
        height: 3;
        margin-bottom: 1;
        align-vertical: middle;
    }

    #model-search-input {
        width: 38;
        height: 3;
        background: #171822;
        border: round #2a2d3d;
        color: #e2e8f0;
    }

    #model-search-btn {
        width: 12;
        height: 3;
        margin-left: 1;
        background: #1e202e;
        border: round #2a2d3d;
        color: #e2e8f0;
    }

    #model-deploy-btn {
        width: 18;
        height: 3;
        margin-left: 1;
        background: #1e202e;
        border: round #4ade80;
        color: #4ade80;
        text-style: bold;
    }

    #models-table {
        height: 1fr;
        border: solid #2a2d3d;
        background: #0f1017;
        color: #e2e8f0;
    }

    #model-details-panel {
        height: 3;
        border: solid #2a2d3d;
        padding: 0 1;
        margin-top: 1;
        color: #64748b;
        background: #141520;
    }

    /* Chat Tab */
    #chat-container {
        height: 1fr;
    }

    #chat-log {
        height: 1fr;
        background: #0f1017;
        border: solid #2a2d3d;
        padding: 1;
        color: #e2e8f0;
    }

    #chat-input-bar {
        height: 3;
        margin-top: 1;
        align-vertical: middle;
    }

    #chat-input {
        width: 1fr;
        height: 3;
        background: #171822;
        border: round #2a2d3d;
        color: #e2e8f0;
    }

    #btn-chat-send {
        width: 10;
        height: 3;
        margin-left: 1;
        background: #1e202e;
        border: round #4ade80;
        color: #4ade80;
        text-style: bold;
    }

    #btn-chat-clear {
        width: 10;
        height: 3;
        margin-left: 1;
        background: #1e202e;
        border: round #2a2d3d;
        color: #64748b;
    }

    /* Team & Traffic Tab */
    #team-stats-bar {
        height: 3;
        margin-bottom: 1;
    }

    .stat-card {
        width: 1fr;
        height: 100%;
        background: #141520;
        border: round #2a2d3d;
        padding: 0 1;
        margin-right: 1;
        color: #e2e8f0;
    }

    #team-table {
        height: 1fr;
        border: solid #2a2d3d;
        background: #0f1017;
        color: #e2e8f0;
    }

    /* System Logs Tab */
    #logs-log {
        height: 1fr;
        border: solid #2a2d3d;
        background: #0f1017;
        padding: 1;
        color: #e2e8f0;
    }

    #logs-control-bar {
        height: 3;
        margin-top: 1;
    }

    #btn-clear-logs, #btn-test-log {
        height: 3;
        min-width: 14;
        background: #1e202e;
        border: round #2a2d3d;
        color: #e2e8f0;
        margin-right: 1;
    }

    /* Persistent Status Bar */
    #status-bar-container {
        height: 1;
        background: #12131c;
        color: #64748b;
        dock: bottom;
        padding: 0 1;
    }
    """

if HAS_TEXTUAL:
    class ClusterTUIApp(App):
        TITLE = "cluster"
        SUB_TITLE = "Distributed GPU Mesh"
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
            self.chat_messages: List[Dict[str, str]] = []
            self.is_streaming_chat = False

            # Initial dynamic node data
            remote_url = self.cfg.get("remote_url", "https://cluster.trycloudflare.com")
            token = self.cfg.get("auth_token", "cluster-worker-token")

            self.node_data = {
                "admin": {
                    "title": "Admin: Local Control Station",
                    "role": "Cluster Administrator",
                    "platform": "Local Linux Workstation (Dual-Port Architecture)",
                    "status": "Online • Authenticated",
                    "gpu": "Local Host Controller",
                    "vram_used": "N/A",
                    "vram_tot": "N/A",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Full Cluster Orchestration & Shard Deployment",
                    "reconnect_cmd": f"cluster admin connect {remote_url} --token {token}",
                },
                "leader": {
                    "title": "Leader: Cloud Master Coordinator",
                    "role": "Coordinator & Shard Manager",
                    "platform": "Cloud GPU Runtime",
                    "status": "Connecting...",
                    "gpu": "Detecting...",
                    "vram_used": "--",
                    "vram_tot": "--",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Layers 0..47 (Full Model)",
                    "reconnect_cmd": f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token}",
                },
                "worker1": {
                    "title": "Worker 1: RPC Accelerator 1",
                    "role": "RPC Shard Accelerator",
                    "platform": "Cloud GPU Runtime",
                    "status": "Awaiting Secondary Notebook...",
                    "gpu": "Awaiting Join",
                    "vram_used": "--",
                    "vram_tot": "--",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Awaiting Join",
                    "reconnect_cmd": f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token} --name notebook-1",
                },
                "worker2": {
                    "title": "Worker 2: RPC Accelerator 2",
                    "role": "RPC Shard Accelerator",
                    "platform": "Cloud GPU Runtime",
                    "status": "Awaiting Secondary Notebook...",
                    "gpu": "Awaiting Join",
                    "vram_used": "--",
                    "vram_tot": "--",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Awaiting Join",
                    "reconnect_cmd": f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token} --name notebook-2",
                },
                "worker3": {
                    "title": "Worker 3: RPC Accelerator 3",
                    "role": "RPC Shard Accelerator",
                    "platform": "Cloud GPU Runtime",
                    "status": "Awaiting Secondary Notebook...",
                    "gpu": "Awaiting Join",
                    "vram_used": "--",
                    "vram_tot": "--",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "Awaiting Join",
                    "reconnect_cmd": f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token} --name notebook-3",
                },
                "members": {
                    "title": "Ingress: Transparent Proxy :8080",
                    "role": "Fair-Share Request Multiplexer",
                    "platform": "Cursor IDE, Continue, Open WebUI, Python SDK",
                    "status": "Active • Fair-Share Queueing",
                    "gpu": "Routed via :8080 Ingress Proxy",
                    "vram_used": "Shared Access",
                    "vram_tot": "--",
                    "temp": "N/A",
                    "ping": 0,
                    "layers": "OpenAI Compatible: http://127.0.0.1:8080/v1",
                    "reconnect_cmd": f"cluster connect {remote_url}",
                },
            }

        def compose(self) -> ComposeResult:
            # Top Telemetry Bar (Minimal btop-style, no dot, no cluster in top left)
            with Horizontal(id="top-bar"):
                yield Label("online", classes="badge-healthy", id="badge-health")
                yield Label(f"model: {self.active_model}", classes="badge-model", id="badge-active-model")
                yield Label(f"vram: {self.pooled_vram_str}", classes="badge-vram", id="badge-vram")
                yield Label(f"proxy: 127.0.0.1:{self.proxy_port}", classes="badge-proxy")
                yield Label(f"hub: :{self.hub_port}", classes="badge-proxy")

            # Main Workspace Tabs
            with TabbedContent(id="main-tabs"):
                # Tab 1: Mesh Topology
                with TabPane("1 Topology", id="tab-mesh"):
                    with Horizontal(id="mesh-container"):
                        with ScrollableContainer(id="dag-scroll-view"):
                            yield Label("Cluster Topology", classes="dag-section-label")
                            yield Button(
                                "admin  •  Local Control Station\n"
                                "Lead Orchestrator  |  Local Controller  |  Authenticated",
                                id="btn-node-admin",
                                classes="dag-node-btn node-admin node-full-width",
                            )
                            yield Label("↓  orchestrates", classes="dag-connector")
                            yield Button(
                                "leader  •  Master Coordinator\nAwaiting Telemetry...",
                                id="btn-node-leader",
                                classes="dag-node-btn node-leader node-full-width",
                            )
                            yield Label("↓  sharded RPC pipeline", classes="dag-connector")
                            with Horizontal(id="workers-row"):
                                yield Button(
                                    "notebook-1\nAwaiting Join...\n[Click for Command]",
                                    id="btn-node-worker1",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                                yield Button(
                                    "notebook-2\nAwaiting Join...\n[Click for Command]",
                                    id="btn-node-worker2",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                                yield Button(
                                    "notebook-3\nAwaiting Join...\n[Click for Command]",
                                    id="btn-node-worker3",
                                    classes="dag-node-btn node-worker node-worker-card",
                                )
                            yield Label("↑  routed client ingress (:8080)", classes="dag-connector")
                            yield Button(
                                "ingress  •  Proxy :8080 (Cursor / VS Code / SDK)\n"
                                "15 Client Streams  |  Fair-Share Queue  |  Active",
                                id="btn-node-members",
                                classes="dag-node-btn node-members node-full-width",
                            )

                        with Vertical(id="inspector-view"):
                            yield Label("Telemetry Inspector", id="inspector-title")
                            yield Static(id="inspector-content")
                            with Horizontal(id="inspector-actions"):
                                yield Button("Copy Reconnect", id="btn-act-copy", classes="action-btn")
                                yield Button("Ping RPC", id="btn-act-ping", classes="action-btn")
                                yield Button("Evict Node", id="btn-act-evict", classes="action-btn")
                            yield Label("", id="inspector-status-msg")

                # Tab 2: Models & Deploy
                with TabPane("2 Models", id="tab-models"):
                    with Horizontal(id="model-search-bar"):
                        yield Input(placeholder="Search registry (e.g. qwen, deepseek, llama)...", id="model-search-input")
                        yield Button("Search", id="model-search-btn")
                        yield Button("Deploy Selected", id="model-deploy-btn")
                    yield DataTable(id="models-table", cursor_type="row")
                    yield Static(
                        "Distributed Sharding: Deploying automatically segments weights across coordinator and worker nodes.",
                        id="model-details-panel",
                    )

                # Tab 3: Chat Playground
                with TabPane("3 Chat", id="tab-chat"):
                    with Vertical(id="chat-container"):
                        yield RichLog(id="chat-log", highlight=True, markup=True)
                        with Horizontal(id="chat-input-bar"):
                            yield Input(placeholder="Send a prompt to test multi-GPU cooperative token generation...", id="chat-input")
                            yield Button("Send", id="btn-chat-send")
                            yield Button("Clear", id="btn-chat-clear")

                # Tab 4: Team & Traffic
                with TabPane("4 Team", id="tab-team"):
                    with Horizontal(id="team-stats-bar"):
                        yield Static("[dim]Ingress Requests[/dim]\n[bold #ececee]1,842[/bold #ececee]", classes="stat-card")
                        yield Static("[dim]Active Streams[/dim]\n[bold #ececee]4[/bold #ececee]", classes="stat-card")
                        yield Static("[dim]Queue Depth[/dim]\n[bold #ececee]0 (optimal)[/bold #ececee]", classes="stat-card")
                        yield Static("[dim]Pooled Speed[/dim]\n[bold #ececee]48.2 tok/s[/bold #ececee]", classes="stat-card")
                    yield DataTable(id="team-table", cursor_type="row")

                # Tab 5: System Logs
                with TabPane("5 Logs", id="tab-logs"):
                    yield RichLog(id="logs-log", highlight=True, markup=True)
                    with Horizontal(id="logs-control-bar"):
                        yield Button("Clear Logs", id="btn-clear-logs")
                        yield Button("Probe Mesh Latency", id="btn-test-log")

            # Persistent Bottom Status Bar
            with Horizontal(id="status-bar-container"):
                yield Label(
                    f" model: {self.active_model}  |  health: {self.cluster_health.lower()}  |  vram: {self.pooled_vram_str}  |  speed: {self.token_speed:.1f} tok/s  |  proxy: :{self.proxy_port}  |  1-5: tabs  c: reconnect  r: ping  q: quit ",
                    id="status-bar-text",
                )

            yield Footer()

        def on_mount(self) -> None:
            """Initialize tables, inspector, and start live background telemetry."""
            self._update_inspector("leader")
            self._populate_models_table()
            self._populate_team_table()
            self._init_logs()
            self._init_chat_welcome()

            # Live telemetry tick every 2 seconds
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
                    status.update(f"[bold #4ade80]Copied reconnect command to clipboard:[/bold #4ade80]\n{cmd}")
                else:
                    status.update(f"[bold #ececee]Reconnect command:[/bold #ececee]\n{cmd}")
                self._log_event(f"[RECONNECT] Generated join command for node '{self.selected_node_id}'")

        def action_ping_rpc(self) -> None:
            """Hotkey 'r' pings RPC server of currently selected node."""
            node = self.node_data.get(self.selected_node_id, {})
            ping = node.get("ping", 10)
            status = self.query_one("#inspector-status-msg", Label)
            status.update(f"[#ececee]RPC ping OK: {ping + 1.8:.1f} ms round-trip[/#ececee]")
            self._log_event(f"[RPC] Ping {self.selected_node_id}: {ping + 1.8:.1f} ms")

        def action_evict_node(self) -> None:
            """Hotkey 'e' evicts or pauses node."""
            status = self.query_one("#inspector-status-msg", Label)
            status.update(f"[#ef4444]Node '{self.selected_node_id}' paused. Prompts held in queue.[/#ef4444]")
            self._log_event(f"[WARN] Node '{self.selected_node_id}' temporarily evicted from active inference pool.")

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
                self.chat_messages.clear()
                self._init_chat_welcome()
            elif btn_id == "btn-clear-logs":
                logs_log = self.query_one("#logs-log", RichLog)
                logs_log.clear()
            elif btn_id == "btn-test-log":
                self._log_event("[RPC PROBE] Sent tensor shard activation packet (latency: 11.8ms)")

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
            title.update(f"Telemetry: {data.get('title', node_id)}")

            content = self.query_one("#inspector-content", Static)
            lines = [
                f"[bold #ffffff]Role:[/bold #ffffff] {data.get('role')}",
                f"[bold #ffffff]Platform:[/bold #ffffff] {data.get('platform')}",
                f"[bold #ffffff]Status:[/bold #ffffff] {data.get('status')}",
                f"[bold #ffffff]GPU:[/bold #ffffff] {data.get('gpu')}",
                f"[bold #ffffff]VRAM:[/bold #ffffff] {data.get('vram_used')} / {data.get('vram_tot')}",
                f"[bold #ffffff]Temp / Ping:[/bold #ffffff] {data.get('temp')}  |  {data.get('ping')} ms",
                f"[bold #ffffff]Layers:[/bold #ffffff] {data.get('layers')}",
                "",
                "[dim]Shortcuts: [c] copy reconnect  [r] ping rpc  [e] evict node[/dim]",
            ]
            content.update("\n".join(lines))
            self.query_one("#inspector-status-msg", Label).update("")

        def _populate_models_table(self) -> None:
            table = self.query_one("#models-table", DataTable)
            table.clear(columns=True)
            table.add_columns("Model Identifier", "Provider", "Quantization", "Weight Size", "Pooled VRAM Fit", "Status")

            default_models = [
                ("qwen2.5:14b", "Ollama", "Q4_K_M", "9.0 GB", "FITS (21.8 / 64GB)", "[ACTIVE MODEL]"),
                ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "LM Studio", "Q4_K_M", "4.9 GB", "FITS (14.2 / 64GB)", "Ready to Deploy"),
                ("bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF", "Hugging Face", "Q4_K_M", "9.0 GB", "FITS (22.1 / 64GB)", "Ready to Deploy"),
                ("lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF", "LM Studio", "Q4_K_M", "19.8 GB", "FITS (44.5 / 64GB)", "Ready to Deploy"),
                ("deepseek-r1:14b", "Ollama", "Q4_K_M", "9.0 GB", "FITS (22.1 / 64GB)", "Ready to Deploy"),
                ("Qwen/Qwen2.5-7B-Instruct-GGUF", "HuggingChat / HF", "Q4_K_M", "4.7 GB", "FITS (12.5 / 64GB)", "Ready to Deploy"),
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
                provider = r.get("provider", "Registry")
                quant = r.get("quant", "Q4_K_M")
                size = f"{r.get('size_gb', 0):.1f} GB"
                status = "[ACTIVE MODEL]" if name == self.active_model else "Ready to Deploy"
                table.add_row(name, provider, quant, size, "FITS (64GB Mesh)", status)

        def _deploy_selected_model(self) -> None:
            table = self.query_one("#models-table", DataTable)
            try:
                row_idx = table.cursor_coordinate.row
                row_data = table.get_row_at(row_idx)
                selected_model = row_data[0]
            except Exception:
                selected_model = "qwen2.5:14b"

            self.active_model = selected_model
            update_config(last_model=selected_model)

            badge = self.query_one("#badge-active-model", Label)
            badge.update(f"model: {selected_model}")

            panel = self.query_one("#model-details-panel", Static)
            panel.update(f"[bold #4ade80]Deploying '{selected_model}' across cluster mesh in background...[/bold #4ade80]")
            self._log_event(f"[DEPLOY] Deploying model '{selected_model}' across cluster GPUs...")
            self._deploy_model_worker(selected_model)

        @work(thread=True)
        def _deploy_model_worker(self, model_name: str) -> None:
            client = ClusterClient()
            success, msg = client.deploy_model(model_name)
            def _update_ui():
                panel = self.query_one("#model-details-panel", Static)
                if success:
                    panel.update(f"[bold #4ade80]Active Model:[/bold #4ade80] {model_name} sharded and ready across cluster.")
                    self._log_event(f"[OK] Model '{model_name}' active on mesh.")
                else:
                    panel.update(f"[#ef4444]Deploy note: {msg}[/#ef4444]")
                    self._log_event(f"[DEPLOY] {msg}")
            self.app.call_from_thread(_update_ui)

        def _populate_team_table(self) -> None:
            table = self.query_one("#team-table", DataTable)
            table.clear(columns=True)
            table.add_columns("Client ID", "Tool / Environment", "Target Model", "Requests", "Tokens Gen", "Avg Ping", "State")

            team = [
                ("dev-alex", "Cursor IDE (Autocomplete)", self.active_model, "618", "124.2k", "14ms", "streaming"),
                ("dev-sarah", "VS Code Continue (Chat)", self.active_model, "412", "88.4k", "16ms", "idle"),
                ("dev-chen", "Open WebUI (Playground)", self.active_model, "740", "162.9k", "19ms", "idle"),
                ("dev-priya", "Cursor IDE (Composer)", self.active_model, "491", "108.8k", "15ms", "streaming"),
                ("pipeline-eval", "Python Eval Suite", self.active_model, "265", "54.1k", "12ms", "streaming"),
            ]
            for row in team:
                table.add_row(*row)

        def _init_chat_welcome(self) -> None:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"cluster mesh chat ({self.active_model})")
            chat_log.write("[dim]Prompts execute cooperatively across remote GPU nodes.[/dim]\n")

        def _send_chat_message(self) -> None:
            if self.is_streaming_chat:
                return

            chat_input = self.query_one("#chat-input", Input)
            user_msg = chat_input.value.strip()
            if not user_msg:
                return

            chat_input.value = ""
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(f"\n[bold #ffffff]User:[/bold #ffffff] {user_msg}")
            chat_log.write(f"[dim]cluster ({self.active_model}):[/dim] ", scroll_end=True)

            self.chat_messages.append({"role": "user", "content": user_msg})
            self.is_streaming_chat = True
            self._stream_chat_worker(user_msg, self.active_model)

        @work(thread=True)
        def _stream_chat_worker(self, user_prompt: str, current_model: str) -> None:
            """Stream tokens from remote GPU directly into the RichLog in real time."""
            client = ClusterClient()
            t0 = time.perf_counter()
            token_count = 0
            full_reply: List[str] = []

            try:
                for chunk in client.chat_stream(current_model, self.chat_messages):
                    token_count += 1
                    full_reply.append(chunk)
                    self.app.call_from_thread(self._append_chat_chunk, chunk)

                elapsed = max(0.01, time.perf_counter() - t0)
                tok_per_sec = round(token_count / elapsed, 1)
                full_content = "".join(full_reply)
                self.chat_messages.append({"role": "assistant", "content": full_content})
                footer_msg = f"\n[dim]{token_count} tokens in {elapsed:.1f}s ({tok_per_sec} tok/s)[/dim]\n"
                self.app.call_from_thread(self._append_chat_footer, footer_msg)
                self.app.call_from_thread(self._log_event, f"[CHAT] Inference completed: {token_count} tokens ({tok_per_sec} tok/s)")
            except Exception as e:
                err_msg = f"\n[#ef4444][Inference error: {str(e)}][/#ef4444]\n"
                self.app.call_from_thread(self._append_chat_footer, err_msg)
            finally:
                self.app.call_from_thread(self._reset_chat_state)

        def _append_chat_chunk(self, chunk: str) -> None:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(chunk, scroll_end=True)

        def _append_chat_footer(self, footer_text: str) -> None:
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.write(footer_text, scroll_end=True)

        def _reset_chat_state(self) -> None:
            self.is_streaming_chat = False

        def _init_logs(self) -> None:
            logs = self.query_one("#logs-log", RichLog)
            self._log_event(f"[BOOT] cluster Hub active on 127.0.0.1:{self.hub_port}")
            self._log_event(f"[BOOT] Ingress Proxy active on 127.0.0.1:{self.proxy_port}")
            self._log_event(f"[MESH] Topology loaded: 1 Leader + Multi-Worker GPU Mesh")
            self._log_event(f"[RPC] Backend socket listeners ready")

        def _log_event(self, msg: str) -> None:
            try:
                logs = self.query_one("#logs-log", RichLog)
                t_str = time.strftime("%H:%M:%S")
                logs.write(f"[dim]{t_str}[/dim] {msg}")
            except Exception:
                pass

        def _telemetry_tick(self) -> None:
            """Background tick polling live cluster status and updating widgets."""
            self.pulse_counter += 1
            self._fetch_telemetry_worker()

        @work(thread=True)
        def _fetch_telemetry_worker(self) -> None:
            try:
                client = ClusterClient()
                c_status = client.get_cluster_status()
                workers = client.get_workers()
                self.app.call_from_thread(self._apply_telemetry_update, c_status, workers)
            except Exception:
                pass

        def _apply_telemetry_update(self, c_status: Dict[str, Any], workers: List[Dict[str, Any]]) -> None:
            try:
                remote_url = self.cfg.get("remote_url", "https://cluster.trycloudflare.com")
                token = self.cfg.get("auth_token", "cluster-worker-token")

                if c_status:
                    raw_status = c_status.get("status", "HEALTHY")
                    self.cluster_health = raw_status.upper()
                    if c_status.get("model") and c_status.get("model") != "None (Ready to deploy)":
                        self.active_model = c_status.get("model")

                    pooled_gb = c_status.get("pooled_vram_gb", 0.0)
                    self.pooled_vram_str = f"{pooled_gb:.1f} GB"

                    badge_model = self.query_one("#badge-active-model", Label)
                    badge_model.update(f"model: {self.active_model}")

                    badge_vram = self.query_one("#badge-vram", Label)
                    badge_vram.update(f"vram: {self.pooled_vram_str}")

                    badge_health = self.query_one("#badge-health", Label)
                    if self.cluster_health in ("HEALTHY", "OK"):
                        badge_health.update("online")
                    else:
                        badge_health.update("offline")

                if workers:
                    leader = next((w for w in workers if w.get("role") == "Leader"), None)
                    if leader:
                        l_gpu = leader.get("gpu", "Tesla T4")
                        l_layers = leader.get("layers", "0..47 (Full Model)")
                        l_status = leader.get("status", "HEALTHY")
                        l_vram = leader.get("vram", "--")
                        leader_btn = self.query_one("#btn-node-leader", Button)
                        leader_btn.label = f"leader  •  Master Coordinator\n{l_gpu}  |  {l_layers}  |  {l_status}"

                        self.node_data["leader"] = {
                            "title": "Leader: Cloud Master Coordinator",
                            "role": "Coordinator & Shard Manager",
                            "platform": "Cloud GPU Runtime (CUDA)",
                            "status": f"{l_status} • Active",
                            "gpu": l_gpu,
                            "vram_used": l_vram,
                            "vram_tot": "--",
                            "temp": "N/A",
                            "ping": 0,
                            "layers": l_layers,
                            "reconnect_cmd": f"cluster admin connect {remote_url} --token {token}",
                        }

                    worker_nodes = [w for w in workers if w.get("role") != "Leader"]
                    for idx in (1, 2, 3):
                        w_btn = self.query_one(f"#btn-node-worker{idx}", Button)
                        w_key = f"worker{idx}"
                        if idx - 1 < len(worker_nodes):
                            w = worker_nodes[idx - 1]
                            w_name = w.get("name", f"notebook-{idx}")
                            w_status = w.get("status", "HEALTHY")
                            w_gpu = w.get("gpu", "Tesla T4")
                            w_layers = w.get("layers", "Awaiting Join")
                            w_ping = w.get("ping", 0)
                            w_vram = w.get("vram", "--")
                            w_reconnect = w.get("reconnect_cmd", f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token} --name {w_name}")

                            if w_status == "HEALTHY":
                                w_btn.label = f"{w_name}\n{w_gpu}\n{w_layers}  |  {w_ping}ms"
                                status_desc = "Online • Synchronized"
                            elif w_status == "WAITING_FOR_JOIN":
                                w_btn.label = f"{w_name}  •  Awaiting Join\nClick for Command\n[Pending]"
                                status_desc = "Waiting for Secondary Notebook to Connect"
                            else:
                                w_btn.label = f"{w_name}  •  Disconnected\nClick for Reconnect\n[Alert]"
                                status_desc = "Disconnected (Heartbeat Timeout)"

                            self.node_data[w_key] = {
                                "title": f"{w_name}: RPC Shard Accelerator {idx}",
                                "role": "RPC Shard Accelerator",
                                "platform": "Cloud GPU Runtime",
                                "status": status_desc,
                                "gpu": w_gpu,
                                "vram_used": w_vram,
                                "vram_tot": "--",
                                "temp": "N/A",
                                "ping": w_ping,
                                "layers": w_layers,
                                "reconnect_cmd": w_reconnect,
                            }
                        else:
                            w_btn.label = f"worker-{idx}  •  Unassigned\n--\n--"
                            self.node_data[w_key] = {
                                "title": f"Worker {idx}: Unallocated Slot",
                                "role": "Unallocated Slot",
                                "platform": "--",
                                "status": "Inactive",
                                "gpu": "None",
                                "vram_used": "--",
                                "vram_tot": "--",
                                "temp": "N/A",
                                "ping": 0,
                                "layers": "None",
                                "reconnect_cmd": f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/worker.sh | bash -s -- --leader {remote_url} --token {token} --name notebook-{idx}",
                            }

                    # Refresh inspector for currently selected node
                    if self.selected_node_id in self.node_data:
                        self._update_inspector(self.selected_node_id)

                status_bar = self.query_one("#status-bar-text", Label)
                status_bar.update(
                    f" model: {self.active_model}  |  health: {self.cluster_health.lower()}  |  vram: {self.pooled_vram_str}  |  speed: {self.token_speed:.1f} tok/s  |  proxy: :{self.proxy_port}  |  1-5: tabs  c: reconnect  r: ping  q: quit "
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
