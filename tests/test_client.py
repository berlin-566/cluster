import json
import os
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import unittest

from host_it.client import HostItClient

class MockCloudServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    auth_token = "valid-test-token"

    def log_message(self, format, *args):
        pass

    def _send_json(self, status_code, data_bytes):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data_bytes)

    def do_GET(self):
        auth = self.headers.get("Authorization", "")
        if self.path == "/_host_it/health":
            if auth != f"Bearer {self.auth_token}":
                self._send_json(401, b'{"error": "unauthorized"}')
                return
            self._send_json(200, b'{"status": "ok", "service": "host-it"}')
            return

        if self.path == "/_host_it/gpu":
            self._send_json(200, b'{"gpu_name": "Tesla T4", "vram_total_mb": 15360, "vram_used_mb": 2048}')
            return

        if self.path == "/api/tags":
            self._send_json(200, b'{"models": [{"name": "qwen2.5:14b", "size": 9000000000}]}')
            return

        self._send_json(404, b'{"error": "not found"}')

class TestClient(unittest.TestCase):
    def test_client_url_formatting(self):
        c1 = HostItClient(remote_url="my-tunnel.trycloudflare.com/", auth_token="tok")
        self.assertEqual(c1.remote_url, "https://my-tunnel.trycloudflare.com")

        c2 = HostItClient(remote_url="http://localhost:8080/", auth_token="tok")
        self.assertEqual(c2.remote_url, "http://localhost:8080")

    def test_client_without_remote_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = os.environ.get("HOST_IT_CONFIG_DIR")
            os.environ["HOST_IT_CONFIG_DIR"] = tmpdir
            try:
                client = HostItClient(remote_url="", auth_token="")
                res = client.health()
                self.assertEqual(res["status"], "error")
                self.assertIn("No remote URL configured", res["error"])
            finally:
                if orig:
                    os.environ["HOST_IT_CONFIG_DIR"] = orig
                else:
                    os.environ.pop("HOST_IT_CONFIG_DIR", None)

    def test_client_mock_server_interaction(self):
        server = HTTPServer(("127.0.0.1", 0), MockCloudServer)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # Valid token
            client = HostItClient(remote_url=f"http://127.0.0.1:{port}", auth_token="valid-test-token")
            health = client.health()
            self.assertEqual(health["status"], "ok")
            self.assertIn("latency_ms", health)

            gpu = client.gpu_stats()
            self.assertEqual(gpu["gpu_name"], "Tesla T4")
            self.assertEqual(gpu["vram_total_mb"], 15360)

            models = client.list_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["name"], "qwen2.5:14b")

            # Invalid token
            bad_client = HostItClient(remote_url=f"http://127.0.0.1:{port}", auth_token="wrong-token")
            bad_health = bad_client.health()
            self.assertEqual(bad_health["status"], "unauthorized")
        finally:
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    unittest.main()
