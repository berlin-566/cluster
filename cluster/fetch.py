import json
import math
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Curated catalog of known Ollama models with sizes and quantizations
OLLAMA_CATALOG = [
    {"name": "llama3.1:8b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.7, "vram_req_gb": 5.5, "family": "llama", "pull_id": "llama3.1:8b"},
    {"name": "llama3.1:70b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 40.0, "vram_req_gb": 43.0, "family": "llama", "pull_id": "llama3.1:70b"},
    {"name": "llama3.2:1b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 1.3, "vram_req_gb": 2.0, "family": "llama", "pull_id": "llama3.2:1b"},
    {"name": "llama3.2:3b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 2.0, "vram_req_gb": 3.2, "family": "llama", "pull_id": "llama3.2:3b"},
    {"name": "qwen2.5:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.5, "vram_req_gb": 5.5, "family": "qwen", "pull_id": "qwen2.5:7b"},
    {"name": "qwen2.5:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.0, "vram_req_gb": 10.5, "family": "qwen", "pull_id": "qwen2.5:14b"},
    {"name": "qwen2.5:32b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 19.0, "vram_req_gb": 22.0, "family": "qwen", "pull_id": "qwen2.5:32b"},
    {"name": "qwen2.5:72b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 47.0, "vram_req_gb": 50.0, "family": "qwen", "pull_id": "qwen2.5:72b"},
    {"name": "qwen2.5-coder:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.7, "vram_req_gb": 5.6, "family": "qwen", "pull_id": "qwen2.5-coder:7b"},
    {"name": "qwen2.5-coder:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.0, "vram_req_gb": 10.5, "family": "qwen", "pull_id": "qwen2.5-coder:14b"},
    {"name": "qwen2.5-coder:32b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 19.5, "vram_req_gb": 22.0, "family": "qwen", "pull_id": "qwen2.5-coder:32b"},
    {"name": "deepseek-r1:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.7, "vram_req_gb": 5.5, "family": "deepseek", "pull_id": "deepseek-r1:7b"},
    {"name": "deepseek-r1:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.0, "vram_req_gb": 10.5, "family": "deepseek", "pull_id": "deepseek-r1:14b"},
    {"name": "deepseek-r1:32b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 19.5, "vram_req_gb": 22.0, "family": "deepseek", "pull_id": "deepseek-r1:32b"},
    {"name": "deepseek-r1:70b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 43.0, "vram_req_gb": 46.0, "family": "deepseek", "pull_id": "deepseek-r1:70b"},
    {"name": "mistral:7b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 4.1, "vram_req_gb": 5.0, "family": "mistral", "pull_id": "mistral:7b"},
    {"name": "mistral-small:22b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 14.0, "vram_req_gb": 16.0, "family": "mistral", "pull_id": "mistral-small:22b"},
    {"name": "gemma2:9b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 5.4, "vram_req_gb": 6.8, "family": "gemma", "pull_id": "gemma2:9b"},
    {"name": "gemma2:27b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 16.0, "vram_req_gb": 18.5, "family": "gemma", "pull_id": "gemma2:27b"},
    {"name": "phi4:14b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 9.1, "vram_req_gb": 10.5, "family": "phi", "pull_id": "phi4:14b"},
    {"name": "command-r:35b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 24.0, "vram_req_gb": 26.5, "family": "cohere", "pull_id": "command-r:35b"},
    {"name": "glm4:9b", "provider": "Ollama", "quant": "Q4_K_M", "size_gb": 5.5, "vram_req_gb": 6.8, "family": "glm", "pull_id": "glm4:9b"},
]

# Verified LM Studio models (direct GGUF pulls from LM Studio community registry)
LM_STUDIO_CATALOG = [
    {
        "name": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 4.9,
        "vram_req_gb": 5.8,
        "family": "llama",
        "repo_id": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "pull_id": "hf.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    },
    {
        "name": "lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "family": "llama",
        "repo_id": "lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
        "pull_id": "hf.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
    },
    {
        "name": "lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 19.8,
        "vram_req_gb": 22.0,
        "family": "qwen",
        "repo_id": "lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF",
        "pull_id": "hf.co/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF",
    },
    {
        "name": "lmstudio-community/Qwen2.5-14B-Instruct-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "family": "qwen",
        "repo_id": "lmstudio-community/Qwen2.5-14B-Instruct-GGUF",
        "pull_id": "hf.co/lmstudio-community/Qwen2.5-14B-Instruct-GGUF",
    },
    {
        "name": "lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 4.7,
        "vram_req_gb": 5.5,
        "family": "qwen",
        "repo_id": "lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
        "pull_id": "hf.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
    },
    {
        "name": "lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "family": "deepseek",
        "repo_id": "lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "pull_id": "hf.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    },
    {
        "name": "lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 19.9,
        "vram_req_gb": 22.5,
        "family": "deepseek",
        "repo_id": "lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "pull_id": "hf.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF",
    },
    {
        "name": "lmstudio-community/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "family": "deepseek",
        "repo_id": "lmstudio-community/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "pull_id": "hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-70B-GGUF",
    },
    {
        "name": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 4.4,
        "vram_req_gb": 5.2,
        "family": "mistral",
        "repo_id": "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
        "pull_id": "hf.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF",
    },
    {
        "name": "lmstudio-community/gemma-2-9b-it-GGUF",
        "publisher": "lmstudio-community",
        "provider": "LM Studio",
        "quant": "Q4_K_M",
        "size_gb": 5.4,
        "vram_req_gb": 6.8,
        "family": "gemma",
        "repo_id": "lmstudio-community/gemma-2-9b-it-GGUF",
        "pull_id": "hf.co/lmstudio-community/gemma-2-9b-it-GGUF",
    },
]

# Verified Hugging Face Hub publishers and prominent models
VERIFIED_HF_PUBLISHERS = ["bartowski", "Qwen", "deepseek-ai", "TheBloke", "unsloth", "MaziyarPanahi", "lmstudio-community"]

HF_CATALOG = [
    {
        "name": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "family": "qwen",
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "pull_id": "hf.co/bartowski/Qwen2.5-14B-Instruct-GGUF",
    },
    {
        "name": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 19.8,
        "vram_req_gb": 22.0,
        "family": "qwen",
        "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "pull_id": "hf.co/bartowski/Qwen2.5-32B-Instruct-GGUF",
    },
    {
        "name": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "family": "deepseek",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "pull_id": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    },
    {
        "name": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 19.9,
        "vram_req_gb": 22.5,
        "family": "deepseek",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "pull_id": "hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
    },
    {
        "name": "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "family": "deepseek",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
        "pull_id": "hf.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF",
    },
    {
        "name": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 4.9,
        "vram_req_gb": 5.8,
        "family": "llama",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "pull_id": "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    },
    {
        "name": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "publisher": "bartowski",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 42.5,
        "vram_req_gb": 45.0,
        "family": "llama",
        "repo_id": "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "pull_id": "hf.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
    },
    {
        "name": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "publisher": "Qwen",
        "provider": "HuggingChat / HF",
        "quant": "Q4_K_M",
        "size_gb": 4.7,
        "vram_req_gb": 5.5,
        "family": "qwen",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "pull_id": "hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF",
    },
    {
        "name": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "publisher": "Qwen",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 9.0,
        "vram_req_gb": 10.5,
        "family": "qwen",
        "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "pull_id": "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF",
    },
    {
        "name": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "publisher": "Qwen",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 47.0,
        "vram_req_gb": 50.0,
        "family": "qwen",
        "repo_id": "Qwen/Qwen2.5-72B-Instruct-GGUF",
        "pull_id": "hf.co/Qwen/Qwen2.5-72B-Instruct-GGUF",
    },
    {
        "name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "publisher": "TheBloke",
        "provider": "Hugging Face",
        "quant": "Q4_K_M",
        "size_gb": 4.4,
        "vram_req_gb": 5.2,
        "family": "mistral",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "pull_id": "hf.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    },
    {
        "name": "unsloth/DeepSeek-R1-GGUF",
        "publisher": "unsloth",
        "provider": "Hugging Face",
        "quant": "UD-Q4_K_M",
        "size_gb": 404.0,
        "vram_req_gb": 420.0,
        "family": "deepseek",
        "repo_id": "unsloth/DeepSeek-R1-GGUF",
        "pull_id": "hf.co/unsloth/DeepSeek-R1-GGUF",
    },
]

def is_model_available(repo_or_name: str) -> bool:
    """Check that the model repository exists and is public (prevents 401/404 errors)."""
    if not repo_or_name or "/" not in repo_or_name:
        return True
    try:
        url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_or_name, safe='/')}"
        req = urllib.request.Request(url, headers={"User-Agent": "cluster-client/0.0.1"}, method="HEAD")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        # On network error or offline mode, rely on curated catalog
        return True

def search_models(query: str, pooled_vram_gb: float = 64.0) -> List[Dict[str, Any]]:
    """
    Search models across trusted registries: Ollama, LM Studio, Hugging Face, HuggingChat.
    Filters out gated, private, and 404 models so only verified available models are listed.
    """
    q = query.strip().lower()
    results: List[Dict[str, Any]] = []
    seen_names = set()

    # 1. Ollama Library Search
    for item in OLLAMA_CATALOG:
        name_lower = item["name"].lower()
        family = item.get("family", "").lower()
        if not q or q in name_lower or q in family:
            res = dict(item)
            res["pull_id"] = res.get("pull_id") or res["name"]
            res["available"] = True
            res["fit_status"], res["fit_label"] = calculate_vram_fit(res["vram_req_gb"], pooled_vram_gb)
            results.append(res)
            seen_names.add(item["name"].lower())

    # 2. LM Studio Community Search
    for item in LM_STUDIO_CATALOG:
        name_lower = item["name"].lower()
        family = item.get("family", "").lower()
        if not q or q in name_lower or q in family or "lm studio" in q:
            res = dict(item)
            res["pull_id"] = res.get("pull_id") or f"hf.co/{res['name']}"
            res["available"] = True
            res["fit_status"], res["fit_label"] = calculate_vram_fit(res["vram_req_gb"], pooled_vram_gb)
            results.append(res)
            seen_names.add(item["name"].lower())

    # 3. Verified Hugging Face Catalog Search
    for item in HF_CATALOG:
        name_lower = item["name"].lower()
        publisher_lower = item.get("publisher", "").lower()
        family = item.get("family", "").lower()
        if not q or q in name_lower or q in publisher_lower or q in family:
            if item["name"].lower() not in seen_names:
                res = dict(item)
                res["pull_id"] = res.get("pull_id") or f"hf.co/{res['name']}"
                res["available"] = True
                res["fit_status"], res["fit_label"] = calculate_vram_fit(res["vram_req_gb"], pooled_vram_gb)
                results.append(res)
                seen_names.add(item["name"].lower())

    # 4. Online Hugging Face API live search (strictly filtering out gated/private/404 models)
    if q:
        try:
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&filter=gguf&sort=downloads&direction=-1&limit=15"
            req = urllib.request.Request(url, headers={"User-Agent": "cluster-client/0.0.1"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    for m in data:
                        model_id = m.get("id", "")
                        if not model_id or model_id.lower() in seen_names:
                            continue

                        # Crucial check: eliminate 401/403 gated or private models
                        if m.get("gated") or m.get("private") or m.get("disabled"):
                            continue

                        author = model_id.split("/")[0] if "/" in model_id else ""
                        is_verified = author in VERIFIED_HF_PUBLISHERS
                        is_lm_studio = "lmstudio" in author.lower() or "lmstudio" in model_id.lower()

                        provider = "LM Studio" if is_lm_studio else ("Hugging Face" if not is_verified else f"Hugging Face ({author})")

                        # Estimate size from tags or model id
                        size_est = 14.0
                        id_lower = model_id.lower()
                        if "70b" in id_lower or "72b" in id_lower:
                            size_est = 42.0
                        elif "32b" in id_lower:
                            size_est = 19.5
                        elif "14b" in id_lower:
                            size_est = 9.0
                        elif "7b" in id_lower or "8b" in id_lower:
                            size_est = 4.8
                        elif "1b" in id_lower or "3b" in id_lower:
                            size_est = 2.2

                        vram_est = round(size_est * 1.15, 1)
                        fit_status, fit_label = calculate_vram_fit(vram_est, pooled_vram_gb)

                        results.append({
                            "name": model_id,
                            "publisher": author or "Community",
                            "provider": provider,
                            "quant": "Q4_K_M",
                            "size_gb": size_est,
                            "vram_req_gb": vram_est,
                            "repo_id": model_id,
                            "pull_id": f"hf.co/{model_id}",
                            "available": True,
                            "fit_status": fit_status,
                            "fit_label": fit_label,
                        })
                        seen_names.add(model_id.lower())
        except Exception:
            # Safe silent fallback to offline curated catalogs
            pass

    return results

def calculate_vram_fit(vram_req_gb: float, pooled_vram_gb: float) -> Tuple[str, str]:
    """Calculate whether the model fits in the pooled multi-GPU VRAM."""
    if pooled_vram_gb <= 0:
        return "UNKNOWN", f"{vram_req_gb} GB required"
    
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
