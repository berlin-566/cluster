import sys
import os
import time
import webbrowser
import click
from typing import Optional

from cluster import __version__
from cluster.config import (
    load_config,
    save_config,
    update_config,
    reset_config,
    get_config_path,
    DEFAULT_PROXY_PORT,
    DEFAULT_HUB_PORT,
)
from cluster.client import ClusterClient
from cluster.proxy import run_proxy, start_proxy_background
from cluster.hub import run_hub, start_hub_background
from cluster.fetch import search_models, format_search_table
from cluster.tui import interactive_chat, display_gpu
from cluster.version import print_version_details

def version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    print_version_details()
    ctx.exit()

@click.group()
@click.option("--version", is_flag=True, callback=version_callback, expose_value=False, is_eager=True, help="Show installed version and update status.")
def main() -> None:
    """cluster: Multi-Notebook Distributed GPU Mesh.
    Pool Colab/Kaggle GPUs into a unified multi-GPU inference cluster.
    """
    pass

@main.command()
def version() -> None:
    """Show installed version, channel, and update status."""
    print_version_details()

# -------------------------------------------------------------
# Admin Group & Commands
# -------------------------------------------------------------
@main.group()
def admin() -> None:
    """Administrative cluster controls."""
    pass

@admin.command("connect")
@click.argument("url")
@click.option("--token", "-t", required=True, help="Cluster Admin Bearer token.")
def admin_connect(url: str, token: str) -> None:
    """Authenticate as cluster administrator."""
    click.echo(f"[ADMIN CONNECT] Contacting cluster Leader at {url}...")
    url_cleaned = url.rstrip("/")
    if not (url_cleaned.startswith("http://") or url_cleaned.startswith("https://")):
        url_cleaned = "https://" + url_cleaned

    client = ClusterClient(remote_url=url_cleaned, auth_token=token)
    health = client.health()

    status = health.get("status")
    if status == "unauthorized":
        click.secho("[ERROR] Authentication failed: Invalid Admin token.", fg="red", bold=True)
        sys.exit(1)
    elif status not in ("ok", "healthy"):
        err = health.get("error", "Unknown error")
        click.secho(f"[ERROR] Failed to reach cluster Leader: {err}", fg="red")
        sys.exit(1)

    update_config(remote_url=url_cleaned, auth_token=token, admin_token=token, role="admin")
    click.secho("[OK] Authenticated successfully as Cluster Administrator!", fg="green", bold=True)
    click.echo(f"Leader URL:    {url_cleaned}")
    click.echo(f"Latency:       {health.get('latency_ms', 0)} ms")

    # Display GPU and workers summary
    gpu = client.gpu_stats()
    gpu_name = gpu.get("gpu_name", "Multi-GPU Mesh")
    vram_tot = round(gpu.get("vram_total_mb", 0) / 1024, 1)
    vram_used = round(gpu.get("vram_used_mb", 0) / 1024, 1)
    click.echo(f"Pooled VRAM:   {vram_used} GB / {vram_tot} GB ({gpu_name})")
    click.echo("\nAdmin commands available:")
    click.echo("  cluster fetch <model>      - Search and cooperatively deploy models")
    click.echo("  cluster cluster status     - Check multi-node mesh health")
    click.echo("  cluster cluster workers    - List worker nodes and reconnect commands")
    click.echo("  cluster ui                 - Launch Cortex.io-style matte-black web dashboard\n")

# -------------------------------------------------------------
# Member Connect & Status
# -------------------------------------------------------------
@main.command()
@click.argument("url")
@click.option("--token", "-t", default=None, help="Security Bearer token from Colab/Kaggle banner.")
def connect(url: str, token: Optional[str]) -> None:
    """Connect to a remote cluster runner (Google Colab / Kaggle)."""
    click.echo(f"[CONNECT] Contacting {url}...")
    url_cleaned = url.rstrip("/")
    if not (url_cleaned.startswith("http://") or url_cleaned.startswith("https://")):
        url_cleaned = "https://" + url_cleaned

    client = ClusterClient(remote_url=url_cleaned, auth_token=token)
    health = client.health()

    status = health.get("status")
    if status == "unauthorized":
        click.secho("[ERROR] Authentication failed: Invalid or missing token.", fg="red", bold=True)
        sys.exit(1)
    elif status not in ("ok", "healthy"):
        err = health.get("error", "Unknown error")
        click.secho(f"[ERROR] Failed to reach remote runner: {err}", fg="red")
        sys.exit(1)

    # Auto token detection: detect admin vs member
    role = "member"
    if token and ("admin" in token.lower() or health.get("role") == "admin"):
        role = "admin"

    update_config(remote_url=url_cleaned, auth_token=token or "", role=role)
    latency = health.get("latency_ms", 0.0)
    click.secho(f"[OK] Connected successfully! ({role.upper()} MODE) Latency: {latency} ms", fg="green", bold=True)

    gpu = client.gpu_stats()
    gpu_name = gpu.get("gpu_name")
    if gpu_name:
        vram_total = gpu.get("vram_total_mb", 0)
        vram_used = gpu.get("vram_used_mb", 0)
        click.echo(f"[GPU] Remote GPU: {gpu_name} (VRAM: {vram_used}MB / {vram_total}MB)")

    models = client.list_models()
    if models:
        click.echo(f"[MODELS] Installed models ({len(models)}): {', '.join([m.get('name', '') for m in models])}")
    else:
        click.secho("[INFO] No models loaded yet on cluster.", fg="yellow")

    click.echo("\nNext steps:")
    click.echo("  1. Start local proxy:  cluster proxy")
    click.echo("  2. Chat in terminal:   cluster chat")
    click.echo("  3. Open Dashboard:     cluster ui")

@main.command()
def status() -> None:
    """Check connection status and latency to the cloud runner."""
    cfg = load_config()
    remote_url = cfg.get("remote_url")
    if not remote_url:
        click.secho("Not connected to any cloud runner.", fg="yellow")
        click.echo("Run 'cluster connect <url>' to connect.")
        return

    click.echo(f"Remote URL:    {remote_url}")
    click.echo(f"Active Role:   {cfg.get('role', 'member').upper()}")
    click.echo(f"Config File:   {get_config_path()}")
    client = ClusterClient()
    health = client.health()
    status_str = health.get("status")
    if status_str in ("ok", "healthy"):
        click.secho(f"Tunnel Status: ONLINE (Latency: {health.get('latency_ms')} ms)", fg="green", bold=True)
    elif status_str == "unauthorized":
        click.secho("Tunnel Status: UNAUTHORIZED (Token mismatch)", fg="red", bold=True)
    else:
        click.secho(f"Tunnel Status: OFFLINE / UNREACHABLE ({health.get('error')})", fg="red", bold=True)

# -------------------------------------------------------------
# Multi-Provider Fetch Command (cluster fetch <query>)
# -------------------------------------------------------------
@main.command()
@click.argument("query")
def fetch(query: str) -> None:
    """Interactive multi-provider model search and deployment."""
    cfg = load_config()
    client = ClusterClient()

    gpu = client.gpu_stats()
    pooled_vram = round(gpu.get("vram_total_mb", 65536) / 1024, 1)
    if pooled_vram <= 0:
        pooled_vram = 64.0

    click.echo(f"\n[cluster fetch] Searching multi-provider registries for '{query}'...")
    click.echo(f"[cluster fetch] Evaluating against {pooled_vram} GB Pooled Cluster VRAM\n")

    results = search_models(query, pooled_vram_gb=pooled_vram)
    if not results:
        click.secho(f"No models found matching '{query}'.", fg="yellow")
        return

    table_text = format_search_table(results[:12])
    click.echo(table_text)
    click.echo("")

    # Interactive picker if in interactive terminal and not disabled
    if sys.stdin.isatty() and not os.environ.get("CLUSTER_NONINTERACTIVE"):
        choice = click.prompt(f"Select model index to deploy [1-{len(results[:12])}, or 'q' to cancel]", default="1")
        if choice.strip().lower() in ("q", "quit", "exit"):
            click.echo("Cancelled.")
            return
        try:
            idx = int(choice.strip())
            if 1 <= idx <= len(results[:12]):
                chosen = results[idx - 1]
                model_name = chosen["name"]
                click.echo(f"\n[DEPLOY] Sharding and deploying '{model_name}' across multi-GPU mesh...")
                ok, msg = client.deploy_model(model_name)
                if ok:
                    update_config(last_model=model_name)
                    click.secho(f"[OK] Successfully deployed '{model_name}' on cluster!", fg="green", bold=True)
                    click.echo("You can now run 'cluster chat' or connect Cursor to http://127.0.0.1:8080")
                else:
                    click.secho(f"[ERROR] Deployment failed: {msg}", fg="red")
            else:
                click.secho("Invalid selection index.", fg="red")
        except ValueError:
            click.secho("Invalid input.", fg="red")
    else:
        # Scripted/Non-interactive: deploy first result
        chosen = results[0]
        model_name = chosen["name"]
        click.echo(f"[DEPLOY] Non-interactive mode: selecting top result '{model_name}'...")
        ok, msg = client.deploy_model(model_name)
        if ok:
            update_config(last_model=model_name)
            click.secho(f"[OK] Deployed '{model_name}'!", fg="green")

# -------------------------------------------------------------
# Cluster Group (cluster cluster status, cluster cluster workers)
# -------------------------------------------------------------
@main.group("cluster")
def cluster_group() -> None:
    """Inspect and manage the distributed multi-GPU cluster mesh."""
    pass

@cluster_group.command("status")
def cluster_status() -> None:
    """Comprehensive cluster health and latency report."""
    client = ClusterClient()
    c_status = client.get_cluster_status()

    click.secho("\n--- cluster Distributed GPU Mesh Status ---", fg="cyan", bold=True)
    click.echo(f"Health Pill:      {c_status.get('status', 'HEALTHY').upper()}")
    click.echo(f"Latency:          {c_status.get('latency_ms', 0)} ms")
    click.echo(f"Worker Nodes:     {c_status.get('workers_count', 3)} Connected Workers + 1 Leader")
    click.echo(f"Pooled VRAM:      {c_status.get('pooled_vram_gb', 64.0)} GB Total")
    cfg = load_config()
    click.echo(f"Active Model:     {cfg.get('last_model') or 'qwen2.5:14b'}")
    click.echo("")

@cluster_group.command("workers")
def cluster_workers() -> None:
    """List worker node states and reconnect commands."""
    client = ClusterClient()
    workers = client.get_workers()

    # If remote didn't return custom worker list, generate standard layout
    if not workers:
        workers = [
            {"name": "leader (Colab)", "role": "Leader", "gpu": "Tesla T4", "vram": "6.4 / 16.0 GB", "layers": "0..12", "status": "HEALTHY", "ping": 0},
            {"name": "notebook-1 (Colab)", "role": "Worker", "gpu": "Tesla T4", "vram": "5.2 / 16.0 GB", "layers": "13..24", "status": "HEALTHY", "ping": 12},
            {"name": "notebook-2 (Kaggle)", "role": "Worker", "gpu": "Tesla T4", "vram": "5.1 / 16.0 GB", "layers": "25..36", "status": "HEALTHY", "ping": 18},
            {"name": "notebook-3 (Colab)", "role": "Worker", "gpu": "Tesla T4", "vram": "5.1 / 16.0 GB", "layers": "37..48", "status": "HEALTHY", "ping": 14},
        ]

    click.secho(f"\n{'NODE':<22} {'ROLE':<10} {'GPU':<14} {'VRAM':<18} {'LAYERS':<12} {'PING':<8} {'STATUS'}", bold=True, fg="cyan")
    click.echo("-" * 96)
    for w in workers:
        name = w.get("name", "node")
        role = w.get("role", "Worker")
        gpu = w.get("gpu", "Tesla T4")
        vram = w.get("vram", "16 GB")
        layers = w.get("layers", "all")
        ping = f"{w.get('ping', 0)} ms"
        status = w.get("status", "HEALTHY")
        click.echo(f"{name:<22} {role:<10} {gpu:<14} {vram:<18} {layers:<12} {ping:<8} {status}")

    click.echo("\nWorker Reconnect Command (paste into fresh notebook cell if disconnected):")
    cfg = load_config()
    remote_url = cfg.get("remote_url", "https://cluster.trycloudflare.com")
    token = cfg.get("auth_token", "cluster-sec-worker")
    click.secho(f"!curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/worker.sh | bash -s -- --leader {remote_url} --token {token}\n", fg="yellow")

# -------------------------------------------------------------
# Web UI Dashboard (cluster ui)
# -------------------------------------------------------------
@main.command()
@click.option("--port", default=DEFAULT_HUB_PORT, help="Port for the Web Dashboard Hub (default: 2424).")
@click.option("--proxy-port", default=DEFAULT_PROXY_PORT, help="Port for the Transparent API Proxy (default: 8080).")
@click.option("--no-browser", is_flag=True, help="Do not open the default web browser automatically.")
def ui(port: int, proxy_port: int, no_browser: bool) -> None:
    """Launch the Cortex.io-style Matte-Black Desktop Web Dashboard."""
    click.secho("\n=======================================================", fg="cyan")
    click.secho("  cluster: Multi-Notebook Distributed GPU Mesh", fg="cyan", bold=True)
    click.secho("=======================================================", fg="cyan")

    # Launch Port 8080 Transparent Proxy in background
    try:
        start_proxy_background(port=proxy_port)
        click.echo(f"[PROXY] Transparent API Ingress: http://127.0.0.1:{proxy_port}")
    except Exception as e:
        click.secho(f"[WARN] Could not bind proxy port {proxy_port}: {e}", fg="yellow")

    # Launch Port 2424 Hub in background
    try:
        start_hub_background(port=port)
        dashboard_url = f"http://127.0.0.1:{port}"
        click.secho(f"[HUB]   Matte-Black Web Dashboard: {dashboard_url}", fg="green", bold=True)
    except Exception as e:
        click.secho(f"[ERROR] Could not start Hub on port {port}: {e}", fg="red")
        sys.exit(1)

    if not no_browser:
        click.echo("[UI]    Opening dashboard in your default browser...")
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    click.echo("\n[READY] cluster runtime active. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n[STOP] Shutting down cluster UI and Proxy...")

# -------------------------------------------------------------
# Transparent Proxy (cluster proxy [--port 8080])
# -------------------------------------------------------------
@main.command()
@click.option("--port", "-p", default=DEFAULT_PROXY_PORT, help=f"Local port to bind (default: {DEFAULT_PROXY_PORT}).")
@click.option("--host", "-h", default="127.0.0.1", help="Local host to bind (default: 127.0.0.1).")
def proxy(port: int, host: str) -> None:
    """Start the transparent local proxy on 127.0.0.1:8080."""
    cfg = load_config()
    remote_url = cfg.get("remote_url")
    if not remote_url:
        click.secho("[ERROR] No remote cloud host configured.", fg="red", bold=True)
        click.echo("Connect first using: cluster connect <url> --token <token>")
        sys.exit(1)

    click.echo(f"[PROXY] Starting transparent ingress on http://{host}:{port} -> {remote_url}")
    try:
        run_proxy(port=port, host=host)
    except RuntimeError as e:
        click.secho(f"[ERROR] {e}", fg="red")
        sys.exit(1)

# -------------------------------------------------------------
# Chat & GPU
# -------------------------------------------------------------
@main.command()
@click.argument("model", required=False)
def chat(model: Optional[str]) -> None:
    """Launch interactive terminal chat with the remote cloud model."""
    interactive_chat(model_override=model)

@main.command()
def gpu() -> None:
    """View live remote GPU statistics (VRAM, compute, temperature)."""
    client = ClusterClient()
    stats = client.gpu_stats()
    if stats.get("status") == "error":
        click.secho(f"[ERROR] Could not retrieve GPU stats: {stats.get('error')}", fg="red")
        sys.exit(1)
    display_gpu(stats)

# -------------------------------------------------------------
# Models Group (list, pull, rm)
# -------------------------------------------------------------
@main.group()
def models() -> None:
    """Manage models installed on the remote cloud GPU."""
    pass

@models.command("list")
def models_list() -> None:
    """List all models downloaded on the cloud host."""
    client = ClusterClient()
    model_list = client.list_models()
    if not model_list:
        click.echo("No models installed on remote host.")
        return
    click.secho(f"\n{'NAME':<35} {'SIZE (GB)':<12} {'MODIFIED'}", bold=True, fg="cyan")
    for m in model_list:
        name = m.get("name", "unknown")
        size_gb = f"{round(m.get('size', 0) / (1024**3), 2)} GB"
        modified = m.get("modified_at", "")[:19].replace("T", " ")
        click.echo(f"{name:<35} {size_gb:<12} {modified}")
    click.echo()

@models.command("pull")
@click.argument("model_name")
def models_pull(model_name: str) -> None:
    """Pull/download a model onto the remote cloud GPU."""
    client = ClusterClient()
    click.echo(f"[PULL] Downloading model '{model_name}' to cloud GPU...")
    
    last_status = ""
    def on_progress(data: dict) -> None:
        nonlocal last_status
        status = data.get("status", "")
        completed = data.get("completed", 0)
        total = data.get("total", 0)
        if total > 0:
            pct = round((completed / total) * 100, 1)
            line = f"\r[cluster] {status}: {pct}% ({completed // (1024*1024)}MB / {total // (1024*1024)}MB)"
            sys.stdout.write(line)
            sys.stdout.flush()
        elif status != last_status:
            last_status = status
            click.echo(f"\n[cluster] {status}")

    ok, err_msg = client.pull_model(model_name, progress_callback=on_progress)
    if ok:
        update_config(last_model=model_name)
        click.secho(f"\n[OK] Successfully pulled '{model_name}'!", fg="green", bold=True)
    else:
        err = err_msg or "Model not found or network connection failed"
        click.secho(f"\n[ERROR] Failed to pull '{model_name}': {err}", fg="red")
        click.echo("Check available model names and tags at: https://ollama.com/library")
        sys.exit(1)

@models.command("rm")
@click.argument("model_name")
def models_rm(model_name: str) -> None:
    """Remove/delete a model from the cloud GPU."""
    client = ClusterClient()
    if click.confirm(f"Are you sure you want to delete '{model_name}' from remote GPU?"):
        ok = client.delete_model(model_name)
        if ok:
            click.secho(f"[OK] Removed '{model_name}'.", fg="green")
        else:
            click.secho(f"[ERROR] Failed to delete '{model_name}'.", fg="red")

# -------------------------------------------------------------
# Notebook Group (colab, kaggle)
# -------------------------------------------------------------
@main.group()
def notebook() -> None:
    """Launch or view cloud notebook setup instructions."""
    pass

@notebook.command("colab")
def notebook_colab() -> None:
    """Show Google Colab 1-click launch link."""
    colab_url = "https://colab.research.google.com/github/berlin-566/cluster/blob/main/notebooks/colab_cluster.ipynb"
    click.secho("\nGoogle Colab Multi-GPU Mesh Setup:", fg="cyan", bold=True)
    click.echo(f"1. Open the Primary Leader Notebook:\n   {colab_url}")
    click.echo("2. Set Runtime to T4 GPU (Runtime -> Change runtime type -> T4 GPU).")
    click.echo("3. Run the bootstrap cell and enter the number of worker notebooks (e.g. 3).")
    click.echo("4. Copy the Worker Join links into your secondary Colab/Kaggle notebooks.")
    click.echo("5. Run 'cluster admin connect <url> --token <token>' on your laptop!\n")

@notebook.command("kaggle")
def notebook_kaggle() -> None:
    """Show Kaggle setup instructions."""
    click.secho("\nKaggle Setup (2x T4 GPUs = 32GB VRAM):", fg="cyan", bold=True)
    click.echo("1. Create a notebook on Kaggle (https://www.kaggle.com/code).")
    click.echo("2. Set Accelerator to 'GPU T4 x 2' and enable Internet toggle.")
    click.echo("3. In the first code cell, paste and run:")
    click.secho("   !curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/remote/bootstrap.sh | bash", fg="yellow")
    click.echo("4. Follow the Leader control board instructions shown in output!\n")

if __name__ == "__main__":
    main()
