import unittest
import time
from remote.server import cluster_state, make_ollama_model_dict, calculate_shards

class TestRemoteServer(unittest.TestCase):
    def setUp(self):
        with cluster_state["lock"]:
            cluster_state["workers"].clear()
            cluster_state["health"] = "HEALTHY"

    def tearDown(self):
        with cluster_state["lock"]:
            cluster_state["workers"].clear()

    def test_make_ollama_model_dict(self):
        m = make_ollama_model_dict("qwen2.5:14b")
        self.assertEqual(m["name"], "qwen2.5:14b")
        self.assertEqual(m["model"], "qwen2.5:14b")
        self.assertIn("details", m)
        self.assertEqual(m["details"]["family"], "qwen2")
        self.assertEqual(m["details"]["parameter_size"], "14B")

    def test_reentrant_lock_and_shards_zero_workers(self):
        """Verify calling calculate_shards while holding the lock does not deadlock."""
        with cluster_state["lock"]:
            calculate_shards(total_layers=48)
            self.assertEqual(cluster_state["leader_layers"], "0..47 (Full Model)")

    def test_dynamic_sharding_one_worker(self):
        """When 1 worker joins out of 3 expected, leader and worker divide layers dynamically."""
        with cluster_state["lock"]:
            cluster_state["workers"]["notebook-1"] = {
                "name": "notebook-1",
                "gpu_name": "Tesla T4",
                "status": "HEALTHY",
            }
            cluster_state["workers"]["notebook-2"] = {
                "name": "notebook-2",
                "gpu_name": "Awaiting Worker Node",
                "status": "WAITING_FOR_JOIN",
            }
            cluster_state["workers"]["notebook-3"] = {
                "name": "notebook-3",
                "gpu_name": "Awaiting Worker Node",
                "status": "WAITING_FOR_JOIN",
            }
            calculate_shards(total_layers=48)

            self.assertEqual(cluster_state["leader_layers"], "0..23 (Embeddings & Attention)")
            self.assertEqual(cluster_state["workers"]["notebook-1"]["layers"], "24..47 (LM Head)")
            self.assertEqual(cluster_state["workers"]["notebook-2"]["layers"], "Awaiting Join")
            self.assertEqual(cluster_state["workers"]["notebook-3"]["layers"], "Awaiting Join")

    def test_dynamic_sharding_two_workers(self):
        """When 2 workers join, layers divide across 3 nodes."""
        with cluster_state["lock"]:
            cluster_state["workers"]["notebook-1"] = {
                "name": "notebook-1",
                "status": "HEALTHY",
            }
            cluster_state["workers"]["notebook-2"] = {
                "name": "notebook-2",
                "status": "HEALTHY",
            }
            cluster_state["workers"]["notebook-3"] = {
                "name": "notebook-3",
                "status": "WAITING_FOR_JOIN",
            }
            calculate_shards(total_layers=48)

            self.assertEqual(cluster_state["leader_layers"], "0..15 (Embeddings & Attention)")
            self.assertEqual(cluster_state["workers"]["notebook-1"]["layers"], "16..31 (Intermediate)")
            self.assertEqual(cluster_state["workers"]["notebook-2"]["layers"], "32..47 (LM Head)")
            self.assertEqual(cluster_state["workers"]["notebook-3"]["layers"], "Awaiting Join")

    def test_dynamic_sharding_three_workers(self):
        """When all 3 workers join, layers divide across 4 nodes."""
        with cluster_state["lock"]:
            cluster_state["workers"]["notebook-1"] = {"status": "HEALTHY"}
            cluster_state["workers"]["notebook-2"] = {"status": "HEALTHY"}
            cluster_state["workers"]["notebook-3"] = {"status": "HEALTHY"}
            calculate_shards(total_layers=48)

            self.assertEqual(cluster_state["leader_layers"], "0..11 (Embeddings & Attention)")
            self.assertEqual(cluster_state["workers"]["notebook-1"]["layers"], "12..23 (Intermediate)")
            self.assertEqual(cluster_state["workers"]["notebook-2"]["layers"], "24..35 (Intermediate)")
            self.assertEqual(cluster_state["workers"]["notebook-3"]["layers"], "36..47 (LM Head)")

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
