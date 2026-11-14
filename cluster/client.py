import json
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
import urllib.request
import urllib.error

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from cluster.config import load_config

class ClusterClient:
    def __init__(self, remote_url: Optional[str] = None, auth_token: Optional[str] = None, timeout: float = 120.0):
        cfg = load_config()
        self.remote_url = (remote_url or cfg.get("remote_url") or "").rstrip("/")
        self.auth_token = auth_token or cfg.get("auth_token") or ""
        self.timeout = timeout or float(cfg.get("timeout") or 120.0)

        if self.remote_url and not (self.remote_url.startswith("http://") or self.remote_url.startswith("https://")):
            self.remote_url = "https://" + self.remote_url

    def _get_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": "cluster-client/0.1.0",
            "Content-Type": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def health(self) -> Dict[str, Any]:
        """Check cloud server status and round-trip latency."""
        if not self.remote_url:
            return {"status": "error", "error": "No remote URL configured. Run 'cluster connect <url>' first."}
        
        t0 = time.perf_counter()
        headers = self._get_headers()

        # Support both /_cluster/health and legacy /_host_it/health
        candidates = [f"{self.remote_url}/_cluster/health", f"{self.remote_url}/_host_it/health"]
        last_error = ""

        for url in candidates:
            try:
                if HAS_HTTPX:
                    with httpx.Client(timeout=10.0, follow_redirects=True, trust_env=False) as client:
                        resp = client.get(url, headers=headers)
                        latency = round((time.perf_counter() - t0) * 1000, 1)
                        if resp.status_code == 200:
                            data = resp.json()
                            data["latency_ms"] = latency
                            return data
                        elif resp.status_code == 401:
                            return {"status": "unauthorized", "error": "Invalid auth token. Please check your token.", "latency_ms": latency}
                        elif resp.status_code == 404:
                            continue
                        else:
                            return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text}", "latency_ms": latency}
                else:
                    req = urllib.request.Request(url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        latency = round((time.perf_counter() - t0) * 1000, 1)
                        data = json.loads(resp.read().decode("utf-8"))
                        data["latency_ms"] = latency
                        return data
            except urllib.error.HTTPError as e:
                latency = round((time.perf_counter() - t0) * 1000, 1)
                if e.code == 401:
                    return {"status": "unauthorized", "error": "Invalid auth token.", "latency_ms": latency}
                elif e.code == 404:
                    continue
                return {"status": "error", "error": f"HTTP {e.code}: {e.reason}", "latency_ms": latency}
            except Exception as e:
                last_error = str(e)
                break

        return {"status": "unreachable", "error": last_error or "Endpoint not found", "latency_ms": -1}

    def gpu_stats(self) -> Dict[str, Any]:
        """Fetch remote GPU telemetry (VRAM, utilization, temp)."""
        headers = self._get_headers()
        candidates = [f"{self.remote_url}/_cluster/gpu", f"{self.remote_url}/_host_it/gpu"]
        
        for url in candidates:
            try:
                if HAS_HTTPX:
                    with httpx.Client(timeout=10.0, follow_redirects=True, trust_env=False) as client:
                        resp = client.get(url, headers=headers)
                        if resp.status_code == 200:
                            return resp.json()
                        elif resp.status_code == 404:
                            continue
                        return {"status": "error", "error": resp.text}
                else:
                    req = urllib.request.Request(url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=10.0) as resp:
                        return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                return {"status": "error", "error": str(e)}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        return {"status": "error", "error": "GPU telemetry endpoint not reachable."}

    def list_models(self) -> List[Dict[str, Any]]:
        """List models installed on the remote Ollama instance."""
        url = f"{self.remote_url}/api/tags"
        headers = self._get_headers()
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=15.0, follow_redirects=True, trust_env=False) as client:
                    resp = client.get(url, headers=headers)
                    if resp.status_code == 200:
                        return resp.json().get("models", [])
                    return []
            else:
                req = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("models", [])
        except Exception:
            return []

    def pull_model(self, model_name: str, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[bool, str]:
        """Pull a model on the remote cloud host, streaming status updates."""
        url = f"{self.remote_url}/api/pull"
        headers = self._get_headers()
        body = json.dumps({"name": model_name, "stream": True}).encode("utf-8")
        last_error = ""

        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=httpx.Timeout(self.timeout, read=600.0), follow_redirects=True, trust_env=False) as client:
                    with client.stream("POST", url, headers=headers, content=body) as resp:
                        if resp.status_code != 200:
                            err_text = resp.read().decode("utf-8", errors="ignore")
                            try:
                                err_json = json.loads(err_text)
                                return False, err_json.get("error", err_text)
                            except Exception:
                                return False, f"HTTP {resp.status_code}: {err_text}"
                        for line in resp.iter_lines():
                            if line.strip():
                                try:
                                    status_data = json.loads(line)
                                    if "error" in status_data:
                                        last_error = status_data["error"]
                                    if progress_callback:
                                        progress_callback(status_data)
                                except json.JSONDecodeError:
                                    pass
                if last_error:
                    return False, last_error
                return True, ""
            else:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=600.0) as resp:
                    for line in resp:
                        decoded = line.decode("utf-8").strip()
                        if decoded:
                            try:
                                status_data = json.loads(decoded)
                                if "error" in status_data:
                                    last_error = status_data["error"]
                                if progress_callback:
                                    progress_callback(status_data)
                            except json.JSONDecodeError:
                                pass
                if last_error:
                    return False, last_error
                return True, ""
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                return False, err_data.get("error", str(e))
            except Exception:
                return False, str(e)
        except Exception as e:
            return False, str(e)

    def delete_model(self, model_name: str) -> bool:
        """Delete a model on the remote host."""
        url = f"{self.remote_url}/api/delete"
        headers = self._get_headers()
        body = json.dumps({"name": model_name}).encode("utf-8")
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=20.0, follow_redirects=True, trust_env=False) as client:
                    resp = client.request("DELETE", url, headers=headers, content=body)
                    return resp.status_code == 200
            else:
                req = urllib.request.Request(url, data=body, headers=headers, method="DELETE")
                with urllib.request.urlopen(req, timeout=20.0) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def chat_stream(self, model: str, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Stream token responses for chat messages."""
        url = f"{self.remote_url}/api/chat"
        headers = self._get_headers()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        body = json.dumps(payload).encode("utf-8")

        if HAS_HTTPX:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, read=300.0), follow_redirects=True, trust_env=False) as client:
                with client.stream("POST", url, headers=headers, content=body) as resp:
                    if resp.status_code != 200:
                        yield f"[Error: HTTP {resp.status_code}]"
                        return
                    for line in resp.iter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                msg = data.get("message", {})
                                chunk = msg.get("content", "")
                                if chunk:
                                    yield chunk
                            except json.JSONDecodeError:
                                pass
        else:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=300.0) as resp:
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        try:
                            data = json.loads(decoded)
                            msg = data.get("message", {})
                            chunk = msg.get("content", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            pass

# Backward compatibility alias
HostItClient = ClusterClient
