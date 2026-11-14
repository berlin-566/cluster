import os
import tempfile
import unittest
from click.testing import CliRunner

from cluster.cli import main

class TestCliCommands(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_env = os.environ.get("CLUSTER_CONFIG_DIR")
        os.environ["CLUSTER_CONFIG_DIR"] = self.tmpdir.name
        self.runner = CliRunner()

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["CLUSTER_CONFIG_DIR"] = self.orig_env
        else:
            os.environ.pop("CLUSTER_CONFIG_DIR", None)
        self.tmpdir.cleanup()

    def test_version_command(self):
        result = self.runner.invoke(main, ["version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cluster", result.output)

    def test_status_disconnected(self):
        result = self.runner.invoke(main, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Not connected", result.output)

    def test_cluster_status(self):
        result = self.runner.invoke(main, ["cluster", "status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cluster Distributed GPU Mesh Status", result.output)
        self.assertIn("Pooled VRAM", result.output)

    def test_cluster_workers(self):
        result = self.runner.invoke(main, ["cluster", "workers"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("leader (Colab)", result.output)
        self.assertIn("notebook-1", result.output)
        self.assertIn("Worker Reconnect Command", result.output)

    def test_notebook_colab(self):
        result = self.runner.invoke(main, ["notebook", "colab"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Google Colab", result.output)
        self.assertIn("colab_cluster.ipynb", result.output)

    def test_notebook_kaggle(self):
        result = self.runner.invoke(main, ["notebook", "kaggle"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Kaggle Setup", result.output)

    def test_fetch_command_noninteractive(self):
        env = {"CLUSTER_NONINTERACTIVE": "1"}
        result = self.runner.invoke(main, ["fetch", "qwen"], env=env)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[cluster fetch] Searching multi-provider registries", result.output)
        self.assertIn("POOLED VRAM STATUS", result.output)

    def test_serve_command_help(self):
        result = self.runner.invoke(main, ["serve", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Start local cluster API proxy on port 8080", result.output)
        self.assertIn("--ollama-port", result.output)

    def test_proxy_alias_help(self):
        result = self.runner.invoke(main, ["proxy", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cluster serve", result.output)

    def test_serve_command_unconfigured(self):
        result = self.runner.invoke(main, ["serve"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("No remote cloud host configured", result.output)

if __name__ == "__main__":
    unittest.main()
