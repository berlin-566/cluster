import http.client
import json
import os
import tempfile
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import unittest

from cluster.config import update_config
from cluster.proxy import ThreadedHTTPServer, OllamaProxyHandler

class MockUpstreamOllama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        auth = self.headers.get("Authorization", "")
        if self.path == "/api/tags":
            if auth != "Bearer secret-proxy-token":
                self.send_response(401)
                self.send_header("Connection", "close")
                self.end_headers()
                return
            body = b'{"models": [{"name": "llama3.1:8b"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self):
        auth = self.headers.get("Authorization", "")
        if auth != "Bearer secret-proxy-token":
            self.send_response(401)
            self.send_header("Connection", "close")
            self.end_headers()
            return

        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            req = json.loads(body.decode("utf-8"))

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            # Stream chunks
            chunks = [
                b'{"response": "Hello"}',
                b'{"response": " world"}',
                b'{"done": true}',
            ]
            for c in chunks:
                hex_len = f"{len(c):X}\r\n".encode("latin-1")
                self.wfile.write(hex_len + c + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return

        self.send_response(404)
        self.send_header("Connection", "close")
        self.end_headers()

class TestProxy(unittest.TestCase):
    def test_proxy_root_and_unconfigured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.environ.get("CLUSTER_CONFIG_DIR")
            os.environ["CLUSTER_CONFIG_DIR"] = tmpdir
            
            proxy_server = ThreadedHTTPServer(("127.0.0.1", 0), OllamaProxyHandler)
            proxy_port = proxy_server.server_port
            t = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            t.start()

            try:
                conn = http.client.HTTPConnection("127.0.0.1", proxy_port)

                # Root route check
                conn.request("GET", "/")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                body = resp.read().decode("utf-8")
                self.assertIn("Ollama is running", body)

                # When no remote is configured, API requests should return 503
                conn.request("GET", "/api/tags")
                resp503 = conn.getresponse()
                self.assertEqual(resp503.status, 503)
                data = json.loads(resp503.read().decode("utf-8"))
                self.assertIn("not connected to a remote cloud runner", data["error"])
                conn.close()
            finally:
                proxy_server.shutdown()
                proxy_server.server_close()
                if orig:
                    os.environ["CLUSTER_CONFIG_DIR"] = orig
                else:
                    os.environ.pop("CLUSTER_CONFIG_DIR", None)

    def test_proxy_forwarding_and_streaming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.environ.get("CLUSTER_CONFIG_DIR")
            os.environ["CLUSTER_CONFIG_DIR"] = tmpdir

            upstream = HTTPServer(("127.0.0.1", 0), MockUpstreamOllama)
            upstream_port = upstream.server_port
            up_t = threading.Thread(target=upstream.serve_forever, daemon=True)
            up_t.start()

            update_config(remote_url=f"http://127.0.0.1:{upstream_port}", auth_token="secret-proxy-token")

            proxy_server = ThreadedHTTPServer(("127.0.0.1", 0), OllamaProxyHandler)
            proxy_port = proxy_server.server_port
            proxy_t = threading.Thread(target=proxy_server.serve_forever, daemon=True)
            proxy_t.start()

            try:
                conn = http.client.HTTPConnection("127.0.0.1", proxy_port)

                # Verify proxy forward of /api/tags
                conn.request("GET", "/api/tags")
                resp = conn.getresponse()
                self.assertEqual(resp.status, 200)
                tags_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(tags_data["models"][0]["name"], "llama3.1:8b")

                # Verify streaming chunked response of /api/generate
                post_body = json.dumps({"prompt": "hi"}).encode("utf-8")
                conn.request("POST", "/api/generate", body=post_body, headers={"Content-Type": "application/json"})
                resp_stream = conn.getresponse()
                self.assertEqual(resp_stream.status, 200)
                self.assertEqual(resp_stream.getheader("Transfer-Encoding"), "chunked")

                stream_data = resp_stream.read().decode("utf-8")
                self.assertIn("Hello", stream_data)
                self.assertIn("world", stream_data)

                conn.close()
            finally:
                proxy_server.shutdown()
                proxy_server.server_close()
                upstream.shutdown()
                upstream.server_close()
                if orig:
                    os.environ["CLUSTER_CONFIG_DIR"] = orig
                else:
                    os.environ.pop("CLUSTER_CONFIG_DIR", None)

if __name__ == "__main__":
    unittest.main()
