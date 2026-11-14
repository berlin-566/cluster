import json
import os
from pathlib import Path
import subprocess
from cluster import __version__

def get_install_dir() -> Path:
    # 1. Check if running inside installed venv
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path and Path(venv_path).parent.name == "cluster":
        return Path(venv_path).parent
    # 2. Standard location
    std_path = Path.home() / ".local" / "share" / "cluster"
    if std_path.exists():
        return std_path
    # 3. Fallback to current directory repo root
    curr = Path(__file__).resolve().parent.parent
    return curr

def get_local_commit(install_dir: Path) -> str:
    # 1. If git repository is present, check real-time git commit
    if (install_dir / ".git").exists():
        try:
            res = subprocess.check_output(
                ["git", "-C", str(install_dir), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            if res:
                return res
        except Exception:
            pass

    # 2. Try install_info file fallback
    info_file = install_dir / ".install_info"
    if info_file.exists():
        try:
            with open(info_file, "r") as f:
                data = json.load(f)
                commit = data.get("commit")
                if commit:
                    return commit[:7]
        except Exception:
            pass

    return "unknown"

def get_install_channel(install_dir: Path) -> str:
    info_file = install_dir / ".install_info"
    if info_file.exists():
        try:
            with open(info_file, "r") as f:
                data = json.load(f)
                return data.get("channel", "main branch")
        except Exception:
            pass
    return "main branch"

def get_remote_main_commit() -> str:
    try:
        res = subprocess.check_output(
            ["git", "ls-remote", "https://github.com/berlin-566/cluster.git", "refs/heads/main"],
            stderr=subprocess.DEVNULL,
            timeout=3.5,
            text=True
        ).strip()
        if res:
            return res.split()[0][:7]
    except Exception:
        pass
    return ""

def get_version_details() -> dict:
    install_dir = get_install_dir()
    local_commit = get_local_commit(install_dir)
    channel = get_install_channel(install_dir)
    remote_commit = get_remote_main_commit()

    is_up_to_date = None
    if remote_commit and local_commit != "unknown":
        is_up_to_date = (local_commit == remote_commit)

    return {
        "version": __version__,
        "channel": channel,
        "local_commit": local_commit,
        "remote_commit": remote_commit,
        "is_up_to_date": is_up_to_date,
    }

def print_version_details():
    details = get_version_details()
    ver = details["version"]
    channel = details["channel"]
    local_hash = details["local_commit"]
    remote_hash = details["remote_commit"]
    up_to_date = details["is_up_to_date"]

    print(f"cluster version: {ver}")
    print(f"Install Channel: {channel}")
    print(f"Local Commit:    {local_hash}")

    if up_to_date is True:
        print("[STATUS] Up to date via main branch")
    elif up_to_date is False:
        print(f"[STATUS] Not up to date via main branch (remote commit: {remote_hash})")
        print("Update command:  curl -fsSL https://raw.githubusercontent.com/berlin-566/cluster/main/desktop.sh | bash")
    else:
        print("[STATUS] Offline / could not verify remote main branch status")
