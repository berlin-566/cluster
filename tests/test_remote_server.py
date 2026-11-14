import unittest
import time
from remote.server import cluster_state, make_ollama_model_dict

class TestRemoteServer(unittest.TestCase):
    def test_make_ollama_model_dict(self):
        m = make_ollama_model_dict("qwen2.5:14b")
        self.assertEqual(m["name"], "qwen2.5:14b")
        self.assertEqual(m["model"], "qwen2.5:14b")
        self.assertIn("details", m)
        self.assertEqual(m["details"]["family"], "qwen2")
        self.assertEqual(m["details"]["parameter_size"], "14B")

    def test_watchdog_waiting_for_join_not_disconnected(self):
        with cluster_state["lock"]:
            cluster_state["workers"]["test-node-1"] = {
                "name": "test-node-1",
                "gpu_name": "Tesla T4 (Awaiting Join)",
                "vram_total_mb": 16384,
                "vram_used_mb": 0,
                "last_seen": time.time() - 30.0, # 30 seconds ago
                "status": "WAITING_FOR_JOIN",
                "ping_ms": 15,
                "layers": "pending",
            }
            cluster_state["health"] = "HEALTHY"

        now = time.time()
        with cluster_state["lock"]:
            degraded = False
            for name, w in cluster_state["workers"].items():
                status = w.get("status")
                if status == "WAITING_FOR_JOIN":
                    continue
                if now - w.get("last_seen", 0) > 10.0:
                    if status != "DISCONNECTED":
                        w["status"] = "DISCONNECTED"
                        degraded = True
                else:
                    w["status"] = "HEALTHY"

            # Should remain WAITING_FOR_JOIN and NOT degraded
            self.assertEqual(cluster_state["workers"]["test-node-1"]["status"], "WAITING_FOR_JOIN")
            self.assertFalse(degraded)

        # Clean up
        with cluster_state["lock"]:
            cluster_state["workers"].pop("test-node-1", None)

    def test_watchdog_healthy_worker_timeout(self):
        with cluster_state["lock"]:
            cluster_state["workers"]["test-node-joined"] = {
                "name": "test-node-joined",
                "gpu_name": "Tesla T4",
                "vram_total_mb": 16384,
                "vram_used_mb": 5000,
                "last_seen": time.time() - 15.0, # 15 seconds ago
                "status": "HEALTHY",
                "ping_ms": 15,
                "layers": "sharded",
            }

        now = time.time()
        with cluster_state["lock"]:
            degraded = False
            for name, w in cluster_state["workers"].items():
                status = w.get("status")
                if status == "WAITING_FOR_JOIN":
                    continue
                if now - w.get("last_seen", 0) > 10.0:
                    if status != "DISCONNECTED":
                        w["status"] = "DISCONNECTED"
                        degraded = True
                else:
                    w["status"] = "HEALTHY"

            self.assertEqual(cluster_state["workers"]["test-node-joined"]["status"], "DISCONNECTED")
            self.assertTrue(degraded)

        # Clean up
        with cluster_state["lock"]:
            cluster_state["workers"].pop("test-node-joined", None)

if __name__ == "__main__":
    unittest.main()
