import sys
import os
import time
import click
from typing import Optional

from host_it import __version__
from host_it.config import load_config, save_config, update_config, reset_config, get_config_path
from host_it.client import HostItClient
from host_it.proxy import run_proxy, start_proxy_background
from host_it.tui import interactive_chat, display_gpu

@click.group()
@click.version_option(version=__version__, prog_name="host-it")
def main() -> None:
    """host-it: Cloud-to-Local LLM Bridge.
    Host Ollama models on Colab / Kaggle and access them seamlessly on your local machine.
    """
    pass

@main.command()
@click.argument("url")
@click.option("--token", "-t", default=None, help="Security Bearer token from Colab/Kaggle banner.")
def connect(url: str, token: Optional[str]) -> None:
    """Connect to a remote cloud runner (Google Colab / Kaggle)."""
    click.echo(f"[CONNECT] Contacting {url}...")
    
    # Clean URL
    url_cleaned = url.rstrip("/")
    if not (url_cleaned.startswith("http://") or url_cleaned.startswith("https://")):
        url_cleaned = "https://" + url_cleaned

    client = HostItClient(remote_url=url_cleaned, auth_token=token)
    health = client.health()

    status = health.get("status")
    if status == "unauthorized":
        click.secho("[ERROR] Authentication failed: Invalid or missing token.", fg="red", bold=True)
        click.echo("Please provide the token displayed in your Colab/Kaggle runner via --token <token>.")
        sys.exit(1)
    elif status not in ("ok", "healthy"):
        err = health.get("error", "Unknown error")
        click.secho(f"[ERROR] Failed to reach remote runner: {err}", fg="red")
        click.echo("Ensure your Colab / Kaggle notebook is running and the Cloudflare tunnel is active.")
        sys.exit(1)

    # Save to config
    update_config(remote_url=url_cleaned, auth_token=token or "")
    latency = health.get("latency_ms", 0.0)
    click.secho(f"[OK] Connected successfully! Latency: {latency} ms", fg="green", bold=True)

    # Check GPU info
    gpu = client.gpu_stats()
    gpu_name = gpu.get("gpu_name")
    if gpu_name:
        vram_total = gpu.get("vram_total_mb", 0)
        vram_used = gpu.get("vram_used_mb", 0)
        click.echo(f"[GPU] Remote GPU: {gpu_name} (VRAM: {vram_used}MB / {vram_total}MB)")

    # Check available models
    models = client.list_models()
    if models:
        click.echo(f"[MODELS] Installed models ({len(models)}): {', '.join([m.get('name', '') for m in models])}")
    else:
        click.secho("[INFO] No models installed yet on the cloud GPU.", fg="yellow")
        click.echo("Pull your first model using: host-it models pull <model_name>")
        click.echo("Example: host-it models pull llama3.1:8b or host-it models pull qwen2.5:14b")

    click.echo("\nNext steps:")
    click.echo("  1. Start local proxy:  host-it proxy")
    click.echo("  2. Chat in terminal:   host-it chat")

@main.command()
@click.option("--port", "-p", default=11434, help="Local port to bind (default: 11434).")
@click.option("--host", "-h", default="127.0.0.1", help="Local host to bind (default: 127.0.0.1).")
def proxy(port: int, host: str) -> None:
    """Start the transparent local proxy on 127.0.0.1:11434."""
    cfg = load_config()
    remote_url = cfg.get("remote_url")
    if not remote_url:
        click.secho("[ERROR] No remote cloud host configured.", fg="red", bold=True)
        click.echo("Connect first using: host-it connect <url> --token <token>")
        sys.exit(1)

    try:
        run_proxy(port=port, host=host)
    except RuntimeError as e:
        click.secho(f"[ERROR] {e}", fg="red")
        sys.exit(1)

@main.command()
@click.argument("model", required=False)
def chat(model: Optional[str]) -> None:
    """Launch interactive terminal chat with the remote cloud model."""
    interactive_chat(model_override=model)

@main.group()
def models() -> None:
    """Manage models installed on the remote cloud GPU."""
    pass

@models.command("list")
def models_list() -> None:
    """List all models downloaded on the cloud host."""
    client = HostItClient()
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
    client = HostItClient()
    click.echo(f"[PULL] Downloading model '{model_name}' to cloud GPU...")
    
    last_status = ""
    def on_progress(data: dict) -> None:
        nonlocal last_status
        status = data.get("status", "")
        completed = data.get("completed", 0)
        total = data.get("total", 0)
        if total > 0:
            pct = round((completed / total) * 100, 1)
            line = f"\r[host-it] {status}: {pct}% ({completed // (1024*1024)}MB / {total // (1024*1024)}MB)"
            sys.stdout.write(line)
            sys.stdout.flush()
        elif status != last_status:
            last_status = status
            click.echo(f"\n[host-it] {status}")

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
    client = HostItClient()
    if click.confirm(f"Are you sure you want to delete '{model_name}' from remote GPU?"):
        ok = client.delete_model(model_name)
        if ok:
            click.secho(f"[OK] Removed '{model_name}'.", fg="green")
        else:
            click.secho(f"[ERROR] Failed to delete '{model_name}'.", fg="red")

@main.command()
def gpu() -> None:
    """View live remote GPU statistics (VRAM, compute, temperature)."""
    client = HostItClient()
    stats = client.gpu_stats()
    if stats.get("status") == "error":
        click.secho(f"[ERROR] Could not retrieve GPU stats: {stats.get('error')}", fg="red")
        sys.exit(1)
    display_gpu(stats)

@main.command()
def status() -> None:
    """Check connection status and latency to the cloud runner."""
    cfg = load_config()
    remote_url = cfg.get("remote_url")
    if not remote_url:
        click.secho("Not connected to any cloud runner.", fg="yellow")
        click.echo("Run 'host-it connect <url>' to connect.")
        return

    click.echo(f"Remote URL:    {remote_url}")
    click.echo(f"Config File:   {get_config_path()}")
    client = HostItClient()
    health = client.health()
    status_str = health.get("status")
    if status_str in ("ok", "healthy"):
        click.secho(f"Tunnel Status: ONLINE (Latency: {health.get('latency_ms')} ms)", fg="green", bold=True)
    elif status_str == "unauthorized":
        click.secho("Tunnel Status: UNAUTHORIZED (Token mismatch)", fg="red", bold=True)
    else:
        click.secho(f"Tunnel Status: OFFLINE / UNREACHABLE ({health.get('error')})", fg="red", bold=True)

@main.group()
def notebook() -> None:
    """Launch or view cloud notebook setup instructions."""
    pass

@notebook.command("colab")
def notebook_colab() -> None:
    """Show Google Colab 1-click launch link."""
    colab_url = "https://colab.research.google.com/github/berlin-566/host-it/blob/main/notebooks/colab_host_it.ipynb"
    click.secho("\nGoogle Colab Setup:", fg="cyan", bold=True)
    click.echo(f"1. Open the 1-click Colab Notebook:\n   {colab_url}")
    click.echo("2. Change Runtime to GPU (Runtime -> Change runtime type -> T4 GPU).")
    click.echo("3. Run the notebook cell.")
    click.echo("4. Copy the 'host-it connect ...' command shown in the output!\n")

@notebook.command("kaggle")
def notebook_kaggle() -> None:
    """Show Kaggle setup instructions."""
    click.secho("\nKaggle Setup (2x T4 GPUs = 32GB VRAM):", fg="cyan", bold=True)
    click.echo("1. Create a new notebook on Kaggle (https://www.kaggle.com/code).")
    click.echo("2. In the right panel, set Accelerator to 'GPU T4 x 2'.")
    click.echo("3. IMPORTANT: Turn 'Internet' toggle ON (requires verified phone number).")
    click.echo("4. In the first code cell, paste and run:")
    click.secho("   !curl -fsSL https://raw.githubusercontent.com/berlin-566/host-it/main/remote/bootstrap.sh | bash", fg="yellow")
    click.echo("5. Copy the 'host-it connect ...' command shown in the output!\n")

if __name__ == "__main__":
    main()
