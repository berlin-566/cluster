import sys
import time
from typing import List, Dict, Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from cluster.client import ClusterClient
from cluster.config import load_config, update_config

# ANSI fallback colors
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_RED = "\033[31m"
COLOR_DIM = "\033[2m"
COLOR_MAGENTA = "\033[35m"

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

    # Verify connection
    health = client.health()
    if health.get("status") not in ("ok", "healthy"):
        err = health.get("error", "Remote host is unreachable.")
        print(f"{COLOR_RED}[ERROR] Cannot connect to cloud runner: {err}{COLOR_RESET}")
        print("Run 'cluster connect <url>' to set up connection.")
        return

    latency = float(health.get("latency_ms", 0.0))
    gpu_info = client.gpu_stats()

    # Determine model
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

        # Handle slash commands
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

        # Chat completion streaming
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
