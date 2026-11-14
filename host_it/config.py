import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_PORT = 11434
DEFAULT_TIMEOUT = 120

def get_config_dir() -> Path:
    custom_dir = os.environ.get("HOST_IT_CONFIG_DIR")
    if custom_dir:
        path = Path(custom_dir)
    else:
        path = Path.home() / ".config" / "host-it"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_config_path() -> Path:
    return get_config_dir() / "config.json"

def default_config() -> Dict[str, Any]:
    return {
        "remote_url": "",
        "auth_token": "",
        "local_port": DEFAULT_PORT,
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
            return merged
    except Exception:
        return default_config()

def save_config(config: Dict[str, Any]) -> None:
    path = get_config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def update_config(**kwargs: Any) -> Dict[str, Any]:
    cfg = load_config()
    for k, v in kwargs.items():
        if v is not None:
            cfg[k] = v
    save_config(cfg)
    return cfg

def reset_config() -> Dict[str, Any]:
    cfg = default_config()
    save_config(cfg)
    return cfg
