#!/usr/bin/env python3
"""
host-it Remote Cloud Supervisor
Runs inside Google Colab or Kaggle.
Zero external pip dependencies required - uses Python standard library.
"""

import http.client
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading

INTERNAL_OLLAMA_PORT = 11435
PROXY_PORT = 8080

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

def ensure_system_dependencies():
    """Ensure required system tools like zstd and curl are available."""
    if not shutil.which("zstd"):
        print("[SETUP] Required tool 'zstd' is missing. Installing...")
        if shutil.which("apt-get"):
            subprocess.run(
                "DEBIAN_FRONTEND=noninteractive apt-get update -qq && apt-get install -y -qq zstd curl wget procps",
                shell=True,
                check=False
            )
        elif shutil.which("dnf"):
            subprocess.run("dnf install -y -q zstd curl wget procps-ng", shell=True, check=False)
        elif shutil.which("pacman"):
            subprocess.run("pacman -Sy --noconfirm zstd curl wget", shell=True, check=False)

def get_gpu_info():
    """Detect GPU info using nvidia-smi."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        lines = out.splitlines()
        if not lines:
            return {"gpu_name": "No GPU detected", "vram_total_mb": 0, "vram_used_mb": 0}
        
        # In multi-gpu systems (like Kaggle 2x T4), sum total VRAM
        gpus = []
        total_vram = 0
        used_vram = 0
        names = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, tot, used, util, temp = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                gpus.append({"name": name, "total": tot, "used": used, "util": util, "temp": temp})
                total_vram += tot
                used_vram += used
                names.append(name)

        primary_name = f"{len(gpus)}x {names[0]}" if len(gpus) > 1 else names[0]
        return {
            "gpu_name": primary_name,
            "gpu_count": len(gpus),
            "vram_total_mb": total_vram,
            "vram_used_mb": used_vram,
            "gpu_util_pct": gpus[0]["util"] if gpus else 0,
            "temp_c": gpus[0]["temp"] if gpus else 0,
            "gpus": gpus,
        }
    except Exception:
        return {"gpu_name": "CPU Mode / Emulated", "vram_total_mb": 0, "vram_used_mb": 0}

def get_recommended_models(vram_mb):
    vram_gb = vram_mb / 1024.0
    if vram_gb >= 28:
        return ["qwen2.5:32b", "deepseek-r1:32b", "command-r:35b", "llama3.1:8b"]
    elif vram_gb >= 14:
        return ["qwen2.5:14b", "deepseek-r1:14b", "llama3.1:8b", "mistral:7b"]
    elif vram_gb >= 7:
        return ["llama3.1:8b", "qwen2.5:7b", "gemma2:9b"]
    else:
        return ["llama3.2:3b", "qwen2.5:3b", "smollm2:1.7b"]

def install_and_start_ollama():
    """Install Ollama if not present, then launch background daemon."""
    ensure_system_dependencies()

    if not shutil.which("ollama"):
        print("[SETUP] Installing Ollama engine...")
        try:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            print("[OK] Ollama installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install Ollama: {e}")
            sys.exit(1)

    print("[SETUP] Starting Ollama engine on internal port 11435...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{INTERNAL_OLLAMA_PORT}"
    env["OLLAMA_KEEP_ALIVE"] = "24h"
    env["OLLAMA_ORIGINS"] = "*"
    env["OLLAMA_FLASH_ATTENTION"] = "1"

    log_file = open("/tmp/ollama.log", "w")
    proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=log_file, stderr=subprocess.STDOUT)

    # Wait for Ollama to become responsive
    max_wait = 25
    for i in range(max_wait):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", INTERNAL_OLLAMA_PORT, timeout=2)
            conn.request("GET", "/api/tags")
            resp = conn.getresponse()
            if resp.status == 200:
                print("[OK] Ollama engine is responsive and ready.")
                conn.close()
                return proc
        except Exception:
            time.sleep(1)

    print("[WARN] Ollama took longer than expected to respond, continuing...")
    return proc

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class AuthProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    auth_token = ""

    def log_message(self, format, *args):
        pass

    def _is_authorized(self):
        auth_hdr = self.headers.get("Authorization", "")
        expected = f"Bearer {self.auth_token}"
        if auth_hdr == expected:
            return True
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if qs.get("token", [""])[0] == self.auth_token:
            return True
        return False

    def _send_unauthorized(self):
        body = json.dumps({"error": "Unauthorized: Missing or invalid Bearer token"}).encode("utf-8")
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", 'Bearer realm="host-it"')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/_host_it/health":
            body = json.dumps({"status": "ok", "service": "host-it", "version": "0.1.0"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if not self._is_authorized():
            self._send_unauthorized()
            return

        if path == "/_host_it/gpu":
            gpu_data = get_gpu_info()
            body = json.dumps(gpu_data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        self._proxy_to_ollama("GET")

    def do_POST(self):
        if not self._is_authorized():
            self._send_unauthorized()
            return
        self._proxy_to_ollama("POST")

    def do_DELETE(self):
        if not self._is_authorized():
            self._send_unauthorized()
            return
        self._proxy_to_ollama("DELETE")

    def do_PUT(self):
        if not self._is_authorized():
            self._send_unauthorized()
            return
        self._proxy_to_ollama("PUT")

    def do_HEAD(self):
        if not self._is_authorized():
            self._send_unauthorized()
            return
        self._proxy_to_ollama("HEAD")

    def _proxy_to_ollama(self, method):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        conn = http.client.HTTPConnection("127.0.0.1", INTERNAL_OLLAMA_PORT, timeout=300.0)
        try:
            out_headers = {}
            for h, v in self.headers.items():
                if h.lower() not in ("host", "authorization", "content-length"):
                    out_headers[h] = v
            out_headers["Host"] = f"127.0.0.1:{INTERNAL_OLLAMA_PORT}"

            conn.request(method, self.path, body=body, headers=out_headers)
            resp = conn.getresponse()

            self.send_response(resp.status, resp.reason)
            is_chunked = False
            for h, v in resp.getheaders():
                lower_h = h.lower()
                if lower_h == "transfer-encoding" and "chunked" in v.lower():
                    is_chunked = True
                elif lower_h not in HOP_BY_HOP:
                    self.send_header(h, v)
            
            self.send_header("Access-Control-Allow-Origin", "*")

            if is_chunked or resp.getheader("Transfer-Encoding") == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    hex_len = f"{len(chunk):X}\r\n".encode("latin-1")
                    self.wfile.write(hex_len + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            err = json.dumps({"error": f"Ollama engine connection error: {str(e)}"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
        finally:
            conn.close()

def start_auth_proxy(token):
    AuthProxyHandler.auth_token = token
    server = ThreadedHTTPServer(("0.0.0.0", PROXY_PORT), AuthProxyHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def install_and_start_cloudflared():
    """Install cloudflared and create a tunnel pointing to PROXY_PORT."""
    cf_bin = shutil.which("cloudflared") or "/tmp/cloudflared"
    if not os.path.exists(cf_bin) and not shutil.which("cloudflared"):
        print("[SETUP] Downloading Cloudflare Tunnel (cloudflared)...")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        try:
            subprocess.run(f"curl -sSL -o /tmp/cloudflared {url} && chmod +x /tmp/cloudflared", shell=True, check=True)
            cf_bin = "/tmp/cloudflared"
        except Exception as e:
            print(f"[ERROR] Failed to download cloudflared: {e}")
            sys.exit(1)

    print("[TUNNEL] Starting Cloudflare secure tunnel...")
    log_path = "/tmp/cloudflared.log"
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}"],
            stdout=log_f,
            stderr=subprocess.STDOUT
        )

    tunnel_url = None
    pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    for _ in range(30):
        time.sleep(1)
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                content = f.read()
                matches = pattern.findall(content)
                if matches:
                    tunnel_url = matches[0]
                    break

    if not tunnel_url:
        print("[ERROR] Cloudflare tunnel URL not found after 30 seconds.")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                print(f.read())
        sys.exit(1)

    return tunnel_url, proc

def main():
    print("[INIT] Starting host-it Cloud Runner")

    # 1. GPU Detection
    gpu_info = get_gpu_info()
    vram_mb = gpu_info.get("vram_total_mb", 0)
    gpu_name = gpu_info.get("gpu_name", "N/A")
    rec_models = get_recommended_models(vram_mb)

    print(f"[GPU] Detected: {gpu_name} (Total VRAM: {vram_mb // 1024} GB)")
    print(f"[MODELS] Recommended: {', '.join(rec_models)}")

    # 2. Start Ollama
    install_and_start_ollama()

    # 3. Generate Secret Token
    secret_token = os.environ.get("HOST_IT_TOKEN") or f"host-it-sec-{secrets.token_hex(6)}"

    # 4. Start Auth Proxy
    start_auth_proxy(secret_token)
    print(f"[AUTH] Protected proxy active on port {PROXY_PORT}")

    # 5. Start Tunnel
    tunnel_url, tunnel_proc = install_and_start_cloudflared()

    # 6. Display Clean Connection Instructions
    print("\n[READY] host-it Cloud Runner is active")
    print(f"Cloud GPU:      {gpu_name} ({vram_mb // 1024} GB VRAM)")
    print(f"Tunnel URL:     {tunnel_url}")
    print(f"Security Token: {secret_token}")
    print("\nIf host-it is not yet installed on your laptop, install it with:")
    print("curl -fsSL https://raw.githubusercontent.com/berlin-566/host-it/main/desktop.sh | bash")
    print("\nThen run this command on your local machine:")
    print(f"host-it connect {tunnel_url} --token {secret_token}")
    print("\nNext steps on your machine:")
    print("host-it proxy       (start local bridge on http://127.0.0.1:11434)")
    print("host-it chat        (interactive terminal chat)\n")

    # 7. Keepalive Loop
    heartbeat_count = 0
    try:
        while True:
            time.sleep(60)
            heartbeat_count += 1
            if heartbeat_count % 10 == 0:
                print(f"[KEEPALIVE] Session active for {heartbeat_count} minutes. GPU: {gpu_name}")
    except KeyboardInterrupt:
        print("\n[STOP] Stopping host-it cloud runner...")

if __name__ == "__main__":
    main()
