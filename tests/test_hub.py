import http.client
import json
import os
import tempfile
import threading
import time
import unittest

from cluster.config import update_config
from cluster.hub import start_hub_background

class TestHub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.orig_env = os.environ.get("CLUSTER_CONFIG_DIR")
        os.environ["CLUSTER_CONFIG_DIR"] = cls.tmpdir.name

        cls.server = start_hub_background(port=0, host="127.0.0.1")
        cls.port = cls.server.server_address[1]
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        if cls.orig_env is not None:
            os.environ["CLUSTER_CONFIG_DIR"] = cls.orig_env
        else:
            os.environ.pop("CLUSTER_CONFIG_DIR", None)
        cls.tmpdir.cleanup()

    def _get(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read()
        status = resp.status
        headers = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        return status, headers, data

    def _post(self, path, payload_dict):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        body = json.dumps(payload_dict).encode("utf-8")
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        status = resp.status
        conn.close()
        return status, data

    def test_get_dashboard_html(self):
        status, headers, data = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("content-type", ""))
        html = data.decode("utf-8")
        self.assertIn("cluster", html)
        self.assertIn("Live Mesh Topology Graph", html)

    def test_get_cluster_status(self):
        status, _, data = self._get("/api/cluster/status")
        self.assertEqual(status, 200)
        res = json.loads(data.decode("utf-8"))
        self.assertEqual(res.get("health"), "HEALTHY")
        self.assertIn("nodes", res)
        self.assertEqual(len(res["nodes"]), 4)

    def test_get_models_search(self):
        status, _, data = self._get("/api/models/search?q=deepseek")
        self.assertEqual(status, 200)
        res = json.loads(data.decode("utf-8"))
        self.assertIsInstance(res, list)
        self.assertGreater(len(res), 0)

    def test_post_models_deploy(self):
        status, data = self._post("/api/models/deploy", {"model": "qwen2.5:32b"})
        self.assertEqual(status, 200)
        res = json.loads(data.decode("utf-8"))
        self.assertEqual(res.get("status"), "deployed")
        self.assertEqual(res.get("model"), "qwen2.5:32b")

    def test_post_chat(self):
        status, data = self._post("/api/chat", {"messages": [{"role": "user", "content": "test prompt"}]})
        self.assertEqual(status, 200)
        text = data.decode("utf-8")
        self.assertIn("cluster multi-GPU mesh", text)
        self.assertIn("test prompt", text)

if __name__ == "__main__":
    unittest.main()
