import json
import math
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Curated catalog of known Ollama models with sizes and quantizations
OLLAMA_CATALOG = [
    {"name": "llama3.1:8b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.7, "vram_req_gb": 5.5, "family": "llama"},
    {"name": "llama3.1:70b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 40.0, "vram_req_gb": 43.0, "family": "llama"},
    {"name": "llama3.2:1b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 1.3, "vram_req_gb": 2.0, "family": "llama"},
    {"name": "llama3.2:3b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 2.0, "vram_req_gb": 3.2, "family": "llama"},
    {"name": "qwen2.5:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.5, "vram_req_gb": 5.5, "family": "qwen"},
    {"name": "qwen2.5:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.0, "vram_req_gb": 10.5, "family": "qwen"},
    {"name": "qwen2.5:32b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 19.0, "vram_req_gb": 22.0, "family": "qwen"},
    {"name": "qwen2.5:72b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 47.0, "vram_req_gb": 50.0, "family": "qwen"},
    {"name": "deepseek-r1:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.7, "vram_req_gb": 5.5, "family": "deepseek"},
    {"name": "deepseek-r1:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.0, "vram_req_gb": 10.5, "family": "deepseek"},
    {"name": "deepseek-r1:32b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 19.5, "vram_req_gb": 22.0, "family": "deepseek"},
    {"name": "deepseek-r1:70b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 43.0, "vram_req_gb": 46.0, "family": "deepseek"},
    {"name": "mistral:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.1, "vram_req_gb": 5.0, "family": "mistral"},
    {"name": "mistral-small:22b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 14.0, "vram_req_gb": 16.0, "family": "mistral"},
    {"name": "gemma2:9b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 5.4, "vram_req_gb": 6.8, "family": "gemma"},
    {"name": "gemma2:27b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 16.0, "vram_req_gb": 18.5, "family": "gemma"},
    {"name": "phi4:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.1, "vram_req_gb": 10.5, "family": "phi"},
    {"name": "command-r:35b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 24.0, "vram_req_gb": 26.5, "family": "cohere"},
    {"name": "glm4:9b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 5.5, "vram_req_gb": 6.8, "family": "glm"},
]

# Verified Hugging Face Hub publishers and prominent models
VERIFIED_HF_PUBLISHERS = ["bartowski", "Qwen", "deepseek-ai", "TheBloke", "unsloth", "MaziyarPanahi"]

HF_CATALOG = [
    {
        "name": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "publisher": "Qwen",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
    },
    {
        "name": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 19.8,
        "vram_req_gb": 22.0,
        "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "filename": "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 19.9,
        "vram_req_gb": 22.5,
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    },
    {
        "name": "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "filename": "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf",
    },
    {
        "name": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "publisher": "Qwen",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 47.0,
        "vram_req_gb": 50.0,
        "repo_id": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "filename": "qwen2.5-72b-instruct-q4_k_m.gguf",
    },
    {
        "name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "publisher": "TheBloke",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 4.4,
        "vram_req_gb": 5.2,
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
    {
        "name": "unsloth/DeepSeek-R1-GGUF",
        "publisher": "unsloth",
        "provider": "Hugging Face",
        "quant": "UD-Q4_K_M",
        "size_gb": 404.0,
        "vram_req_gb": 420.0,
        "repo_id": "unsloth/DeepSeek-R1-GGUF",
        "filename": "DeepSeek-R1-UD-Q4_K_M.gguf",
    },
]

def search_models(query: str, pooled_vram_gb: float = 64.0) -> List[Dict[str, Any]]:
    """
    Search models across Ollama library and Hugging Face Hub.
    Returns sorted list of matching models with VRAM fit status.
    """
    q = query.strip().lower()
    results: List[Dict[str, Any]] = []

    # 1. Ollama search
    for item in OLLAMA_CATALOG:
        if not q or q in item["name"].lower() or q in item.get("family", ""):
            res = dict(item)
            res["fit_status"], res["fit_label"] = calculate_vram_fit(res["vram_req_gb"], pooled_vram_gb)
            results.append(res)

    # 2. Try online Hugging Face API search
    hf_online_found = False
    if q:
        try:
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&filter=gguf&sort=downloads&direction=-1&limit=15"
            req = urllib.request.Request(url, headers={"User-Agent": "cluster-client/0.0.1"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    for m in data:
                        model_id = m.get("id", "")
                        author = model_id.split("/")[0] if "/" in model_id else ""
                        is_verified = author in VERIFIED_HF_PUBLISHERS
                        # Estimate size from tags or model id
                        size_est = 14.0
                        if "70b" in model_id.lower() or "72b" in model_id.lower():
                            size_est = 42.0
                        elif "32b" in model_id.lower():
                            size_est = 19.5
                        elif "14b" in model_id.lower():
                            size_est = 9.0
                        elif "7b" in model_id.lower() or "8b" in model_id.lower():
                            size_est = 4.8
                        elif "1b" in model_id.lower() or "3b" in model_id.lower():
                            size_est = 2.2

                        vram_est = round(size_est * 1.15, 1)
                        fit_status, fit_label = calculate_vram_fit(vram_est, pooled_vram_gb)

                        results.append({
                            "name": model_id,
                            "publisher": author,
                            "provider": "Hugging Face" if not is_verified else f"Hugging Face ({author})",
                            "quant": "Q4_K_M",
                            "size_gb": size_est,
                            "vram_req_gb": vram_est,
                            "repo_id": model_id,
                            "fit_status": fit_status,
                            "fit_label": fit_label,
                        })
                        hf_online_found = True
        except Exception:
            pass

    # Fallback to local HF Catalog
    if not hf_online_found:
        for item in HF_CATALOG:
            if not q or q in item["name"].lower() or q in item.get("publisher", "").lower():
                res = dict(item)
                res["fit_status"], res["fit_label"] = calculate_vram_fit(res["vram_req_gb"], pooled_vram_gb)
                results.append(res)

    return results

def calculate_vram_fit(vram_req_gb: float, pooled_vram_gb: float) -> Tuple[str, str]:
    """Calculate whether the model fits in the pooled multi-GPU VRAM."""
    if pooled_vram_gb <= 0:
        return "UNKNOWN", f"{vram_req_gb} GB required"
    
    pct = (vram_req_gb / pooled_vram_gb) * 100
    if vram_req_gb <= pooled_vram_gb * 0.85:
        return "FIT", f"Fits ({vram_req_gb} GB / {round(pooled_vram_gb, 1)} GB Pooled)"
    elif vram_req_gb <= pooled_vram_gb:
        return "TIGHT", f"Tight Fit ({vram_req_gb} GB / {round(pooled_vram_gb, 1)} GB Pooled)"
    else:
        return "EXCEEDS", f"Exceeds VRAM ({vram_req_gb} GB > {round(pooled_vram_gb, 1)} GB)"

def format_search_table(results: List[Dict[str, Any]]) -> str:
    """Format model search results as a clean, high-legibility terminal table."""
    lines = []
    lines.append(f"{'#':<4} {'MODEL':<44} {'PROVIDER':<16} {'QUANT':<10} {'SIZE':<9} {'POOLED VRAM STATUS'}")
    lines.append("-" * 110)
    for i, item in enumerate(results, start=1):
        idx_str = f"[{i}]"
        name = item["name"][:42]
        provider = item["provider"][:14]
        quant = item.get("quant", "Q4_K_M")[:8]
        size_str = f"{item['size_gb']} GB"
        fit_status = item.get("fit_status", "FIT")
        fit_label = item.get("fit_label", "")
        status_tag = f"[{fit_status}] {fit_label}"
        lines.append(f"{idx_str:<4} {name:<44} {provider:<16} {quant:<10} {size_str:<9} {status_tag}")
    return "\n".join(lines)
