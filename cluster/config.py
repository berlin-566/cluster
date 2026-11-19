import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PROXY_PORT = 8080
DEFAULT_HUB_PORT = 2424
DEFAULT_PORT = DEFAULT_PROXY_PORT  # Alias for backward compatibility
DEFAULT_TIMEOUT = 120

def get_config_dir() -> Path:
    custom_dir = os.environ.get("CLUSTER_CONFIG_DIR") or os.environ.get("HOST_IT_CONFIG_DIR")
    if custom_dir:
        path = Path(custom_dir)
    else:
        new_path = Path.home() / ".config" / "cluster"
        old_path = Path.home() / ".config" / "host-it"
        if not new_path.exists() and old_path.exists():
            path = old_path
        else:
            path = new_path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path

def get_config_path() -> Path:
    return get_config_dir() / "config.json"

def default_config() -> Dict[str, Any]:
    return {
        "remote_url": "",
        "auth_token": "",
        "admin_token": "",
        "role": "member",  # "admin" or "member"
        "local_port": DEFAULT_PROXY_PORT,
        "proxy_port": DEFAULT_PROXY_PORT,
        "hub_port": DEFAULT_HUB_PORT,
        "theme": "dark",
        "last_model": "",
        "timeout": DEFAULT_TIMEOUT,
    }

def load_config() -> Dict[str, Any]:
    path = get_config_path()
    if not path.exists():
        cfg = default_config()
        save_config(cfg)
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged = default_config()
            merged.update(data)
            # Ensure local_port and proxy_port stay in sync if one was set
            if "local_port" in data and "proxy_port" not in data:
                merged["proxy_port"] = data["local_port"]
            elif "proxy_port" in data:
                merged["local_port"] = data["proxy_port"]
            return merged
    except Exception:
        return default_config()

def save_config(config: Dict[str, Any]) -> None:
    path = get_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except OSError:
        pass

def update_config(**kwargs: Any) -> Dict[str, Any]:
    cfg = load_config()
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
            if k == "proxy_port":
                cfg["local_port"] = v
            elif k == "local_port":
                cfg["proxy_port"] = v
    save_config(cfg)
    return cfg

def reset_config() -> Dict[str, Any]:
    cfg = default_config()
    save_config(cfg)
    return cfg
