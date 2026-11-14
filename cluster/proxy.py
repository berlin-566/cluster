import hashlib
import http.client
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional, Tuple, List, Dict, Any
import urllib.parse

from cluster.config import load_config, DEFAULT_PROXY_PORT, DEFAULT_HUB_PORT
from cluster.client import ClusterClient

logger = logging.getLogger("cluster.proxy")

def make_ollama_model(name: str, modified_at: Optional[str] = None) -> Dict[str, Any]:
    """Format model dictionary to strictly comply with Ollama CLI specification."""
    clean_name = name or "qwen2.5:14b"
    family = "qwen2" if "qwen" in clean_name.lower() else "llama"
    param_size = "14B"
    if ":" in clean_name:
        tag = clean_name.split(":", 1)[1]
        if "b" in tag.lower():
            param_size = tag.upper()
    mod_time = modified_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    digest_hash = hashlib.sha256(clean_name.encode("utf-8")).hexdigest()
    return {
        "name": clean_name,
        "model": clean_name,
        "modified_at": mod_time,
        "size": 9200000000,
        "digest": f"sha256:{digest_hash}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": param_size,
            "quantization_level": "Q4_K_M",
        },
    }

def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local TCP port is free for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)

class OllamaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        pass

    def _get_remote_target(self) -> Tuple[str, str, int, bool, str]:
        cfg = load_config()
        remote_url = (cfg.get("remote_url") or "").rstrip("/")
        auth_token = cfg.get("auth_token") or ""

        if not remote_url:
            return "", "", 0, False, auth_token

        parsed = urllib.parse.urlsplit(remote_url)
        is_https = parsed.scheme == "https"
        host = parsed.hostname or ""
        port = parsed.port or (443 if is_https else 80)
        return remote_url, host, port, is_https, auth_token

    def _send_error_json(self, status_code: int, message: str) -> None:
        try:
            payload = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._proxy_request("HEAD")

    def do_GET(self) -> None:
        if self.path in ("/", ""):
            body = b"Ollama is running (via cluster cloud bridge on port 8080)\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path in ("/_cluster/status", "/_host_it/status"):
            cfg = load_config()
            client = ClusterClient()
            health = client.health()
            payload = json.dumps({
                "cluster": "running",
                "proxy_port": DEFAULT_PROXY_PORT,
                "remote_url": cfg.get("remote_url"),
                "health": health,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # OpenAI compatible models list endpoint
        if self.path in ("/v1/models", "/v1/models/"):
            client = ClusterClient()
            models_list = client.list_models()
            model_items = []
            for m in models_list:
                m_name = m.get("name", "default")
                model_items.append({
                    "id": m_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "cluster"
                })
            payload = json.dumps({"object": "list", "data": model_items}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # Ollama Version endpoint
        if self.path in ("/api/version", "/api/version/"):
            payload = json.dumps({"version": "0.5.4"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # Ollama Running Processes endpoint (ollama ps)
        if self.path in ("/api/ps", "/api/ps/"):
            cfg = load_config()
            active_m = cfg.get("last_model") or "qwen2.5:14b"
            m_obj = make_ollama_model(active_m)
            m_obj["expires_at"] = "2099-01-01T00:00:00Z"
            m_obj["size_vram"] = 9200000000
            payload = json.dumps({"models": [m_obj]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        # Ollama Tags / Models endpoint (ollama list)
        if self.path in ("/api/tags", "/api/tags/"):
            remote_url, host, port, is_https, auth_token = self._get_remote_target()
            if not remote_url or not host:
                self._send_error_json(503, "cluster proxy is not connected to a remote cloud runner. Run 'cluster connect <url>' first.")
                return

            models_list = []
            try:
                client = ClusterClient()
                models_list = client.list_models()
            except Exception:
                models_list = []

            cfg = load_config()
            active_m = cfg.get("last_model") or "qwen2.5:14b"
            if not models_list:
                models_list.append(make_ollama_model(active_m))

            formatted_models = []
            for m in models_list:
                m_name = m.get("name") or m.get("model") or active_m
                if "details" not in m or "size" not in m:
                    formatted_models.append(make_ollama_model(m_name, m.get("modified_at")))
                else:
                    formatted_models.append(m)

            payload = json.dumps({"models": formatted_models}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        self._proxy_request("GET")

    def do_POST(self) -> None:
        # Ollama Model Inspection (ollama run / show)
        if self.path in ("/api/show", "/api/show/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"
            model_req = ""
            try:
                body_json = json.loads(body_bytes.decode("utf-8"))
                model_req = body_json.get("name") or body_json.get("model") or ""
            except Exception:
                pass

            cfg = load_config()
            active_m = model_req or cfg.get("last_model") or "qwen2.5:14b"
            m_obj = make_ollama_model(active_m)
            show_payload = {
                "license": "Apache 2.0",
                "modelfile": f"# Modelfile generated by cluster mesh\nFROM {active_m}\nPARAMETER stop <|im_end|>\n",
                "parameters": "stop <|im_end|>",
                "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
                "details": m_obj["details"],
                "model_info": {
                    "general.architecture": m_obj["details"]["family"],
                },
            }
            payload = json.dumps(show_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        self._proxy_request("POST")

    def do_DELETE(self) -> None:
        self._proxy_request("DELETE")

    def do_PUT(self) -> None:
        self._proxy_request("PUT")

    def _proxy_request(self, method: str) -> None:
        remote_url, host, port, is_https, auth_token = self._get_remote_target()

        if not remote_url or not host:
            self._send_error_json(503, "cluster is not connected to a remote cloud runner. Run 'cluster connect <url>' first.")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        target_path = self.path

        conn: Optional[http.client.HTTPConnection] = None
        try:
            if is_https:
                import ssl
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=600.0)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=600.0)

            out_headers = {}
            for header_name, header_val in self.headers.items():
                if header_name.lower() not in HOP_BY_HOP:
                    out_headers[header_name] = header_val

            # Inject Bearer auth token if missing
            if auth_token and "authorization" not in [h.lower() for h in out_headers]:
                out_headers["Authorization"] = f"Bearer {auth_token}"

            out_headers["Host"] = host

            conn.request(method, target_path, body=body, headers=out_headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)

            is_chunked = False
            for header_name, header_val in resp.getheaders():
                lower_h = header_name.lower()
                if lower_h == "transfer-encoding" and "chunked" in header_val.lower():
                    is_chunked = True
                elif lower_h not in HOP_BY_HOP:
                    self.send_header(header_name, header_val)

            self.send_header("Access-Control-Allow-Origin", "*")

            if is_chunked or resp.getheader("Transfer-Encoding") == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        hex_len = f"{len(chunk):X}\r\n".encode("latin-1")
                        self.wfile.write(hex_len + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.end_headers()
                try:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self._send_error_json(502, f"Failed to connect to remote cluster runner: {str(e)}")
        finally:
            if conn:
                conn.close()

def run_proxy(port: int = DEFAULT_PROXY_PORT, host: str = "127.0.0.1", ollama_port: int = 11434) -> None:
    """Run the transparent proxy HTTP server in foreground with optional Ollama port dual-binding."""
    server = ThreadedHTTPServer((host, port), OllamaProxyHandler)
    logger.info(f"cluster proxy running on http://{host}:{port}")

    ollama_server = None
    if ollama_port and ollama_port != port:
        if is_port_available(ollama_port, host):
            try:
                ollama_server = ThreadedHTTPServer((host, ollama_port), OllamaProxyHandler)
                t_ollama = threading.Thread(target=ollama_server.serve_forever, daemon=True)
                t_ollama.start()
                print(f"[OLLAMA] Dual-bound on http://{host}:{ollama_port} (direct Ollama CLI support active)")
            except Exception as e:
                logger.warning(f"Failed to dual-bind Ollama port {ollama_port}: {e}")
        else:
            print(f"[OLLAMA] Port {ollama_port} is busy. To route Ollama CLI to cluster, run: export OLLAMA_HOST={host}:{port}")

    print(f"[SERVE] API proxy listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping proxy server...")
    finally:
        server.server_close()
        if ollama_server:
            ollama_server.server_close()

def start_proxy_background(port: int = DEFAULT_PROXY_PORT, host: str = "127.0.0.1", ollama_port: int = 11434) -> ThreadedHTTPServer:
    """Launch proxy HTTP server in background thread with optional Ollama dual-binding."""
    server = ThreadedHTTPServer((host, port), OllamaProxyHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    if ollama_port and ollama_port != port and is_port_available(ollama_port, host):
        try:
            ollama_server = ThreadedHTTPServer((host, ollama_port), OllamaProxyHandler)
            t_ollama = threading.Thread(target=ollama_server.serve_forever, daemon=True)
            t_ollama.start()
        except Exception:
            pass

    return server
