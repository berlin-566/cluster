import http.client
import json
import logging
import socket
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from typing import Optional, Tuple
import urllib.parse

from host_it.config import load_config
from host_it.client import HostItClient

logger = logging.getLogger("host-it.proxy")

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

class OllamaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy standard request logs unless debug enabled
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
        base_path = parsed.path.rstrip("/")
        return remote_url, host, port, is_https, auth_token

    def _send_error_json(self, status_code: int, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()

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
        if self.path == "/" or self.path == "":
            # Standard Ollama root response
            body = b"Ollama is running (via host-it cloud bridge)\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/_host_it/status":
            cfg = load_config()
            client = HostItClient()
            health = client.health()
            payload = json.dumps({
                "host_it": "running",
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

        self._proxy_request("GET")

    def do_POST(self) -> None:
        self._proxy_request("POST")

    def do_DELETE(self) -> None:
        self._proxy_request("DELETE")

    def do_PUT(self) -> None:
        self._proxy_request("PUT")

    def _proxy_request(self, method: str) -> None:
        remote_url, host, port, is_https, auth_token = self._get_remote_target()

        if not remote_url or not host:
            self._send_error_json(503, "host-it is not connected to a remote cloud runner. Run 'host-it connect <url>' first.")
            return

        # Read incoming request body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Build outbound path and query
        target_path = self.path

        # Connect to remote target
        conn: Optional[http.client.HTTPConnection] = None
        try:
            if is_https:
                import ssl
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=180.0)
            else:
                conn = http.client.HTTPConnection(host, port, timeout=180.0)

            # Prepare outbound headers
            out_headers = {}
            for header_name, header_val in self.headers.items():
                if header_name.lower() not in HOP_BY_HOP:
                    out_headers[header_name] = header_val

            out_headers["Host"] = host
            if auth_token:
                out_headers["Authorization"] = f"Bearer {auth_token}"

            conn.request(method, target_path, body=body, headers=out_headers)
            upstream_resp = conn.getresponse()

            # Stream response back to client
            self.send_response(upstream_resp.status, upstream_resp.reason)
            
            # Forward headers (excluding hop-by-hop)
            is_chunked = False
            for header, val in upstream_resp.getheaders():
                lower_h = header.lower()
                if lower_h not in HOP_BY_HOP:
                    self.send_header(header, val)
                elif lower_h == "transfer-encoding" and "chunked" in val.lower():
                    is_chunked = True

            self.send_header("Access-Control-Allow-Origin", "*")
            
            if is_chunked or upstream_resp.getheader("Transfer-Encoding") == "chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                
                # Stream chunked
                while True:
                    chunk = upstream_resp.read(4096)
                    if not chunk:
                        break
                    # Send HTTP chunk
                    hex_len = f"{len(chunk):X}\r\n".encode("latin-1")
                    self.wfile.write(hex_len + chunk + b"\r\n")
                    self.wfile.flush()
                # Final zero chunk
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                # Regular stream / content-length response
                self.end_headers()
                while True:
                    chunk = upstream_resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except (socket.timeout, TimeoutError):
            self._send_error_json(504, "Request to remote cloud GPU timed out.")
        except ConnectionRefusedError:
            self._send_error_json(502, f"Could not connect to remote host {host}:{port}.")
        except Exception as e:
            self._send_error_json(502, f"Cloud bridge proxy error: {str(e)}")
        finally:
            if conn:
                conn.close()

def run_proxy(port: int = 11434, host: str = "127.0.0.1") -> None:
    """Run the local transparent proxy synchronously."""
    server_address = (host, port)
    try:
        httpd = ThreadedHTTPServer(server_address, OllamaProxyHandler)
    except OSError as e:
        if e.errno == 98 or "Address already in use" in str(e):
            raise RuntimeError(
                f"Port {port} is already in use by another process (likely a local Ollama daemon).\n"
                f"Stop the local daemon (e.g. 'systemctl stop ollama') or run 'host-it proxy --port <other-port>'."
            )
        raise e

    print(f"✨ host-it local proxy running at http://{host}:{port}")
    print(f"👉 Point Cursor, Continue, Open WebUI, or Ollama CLI to: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

def start_proxy_background(port: int = 11434, host: str = "127.0.0.1") -> Tuple[ThreadedHTTPServer, threading.Thread]:
    """Start local proxy in a background daemon thread."""
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, OllamaProxyHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread
