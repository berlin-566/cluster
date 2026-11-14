import json
import os
from pathlib import Path
import tempfile
import unittest

from host_it.config import (
    load_config,
    save_config,
    update_config,
    reset_config,
    get_config_dir,
    get_config_path,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
)

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_env = os.environ.get("HOST_IT_CONFIG_DIR")
        os.environ["HOST_IT_CONFIG_DIR"] = self.tmpdir.name

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["HOST_IT_CONFIG_DIR"] = self.orig_env
        else:
            os.environ.pop("HOST_IT_CONFIG_DIR", None)
        self.tmpdir.cleanup()

    def test_config_directory_and_path(self):
        cfg_dir = get_config_dir()
        self.assertEqual(cfg_dir, Path(self.tmpdir.name))
        self.assertEqual(get_config_path(), Path(self.tmpdir.name) / "config.json")

    def test_default_config_loading(self):
        cfg = load_config()
        self.assertEqual(cfg["remote_url"], "")
        self.assertEqual(cfg["auth_token"], "")
        self.assertEqual(cfg["local_port"], DEFAULT_PORT)
        self.assertEqual(cfg["timeout"], DEFAULT_TIMEOUT)

    def test_save_and_update_config(self):
        updated = update_config(remote_url="https://test.trycloudflare.com", auth_token="sec-12345")
        self.assertEqual(updated["remote_url"], "https://test.trycloudflare.com")
        self.assertEqual(updated["auth_token"], "sec-12345")

        # Verify disk persistence
        with open(get_config_path(), "r") as f:
            disk_data = json.load(f)
            self.assertEqual(disk_data["remote_url"], "https://test.trycloudflare.com")
            self.assertEqual(disk_data["auth_token"], "sec-12345")

        # Partial update
        partial = update_config(last_model="qwen2.5:14b")
        self.assertEqual(partial["remote_url"], "https://test.trycloudflare.com")
        self.assertEqual(partial["last_model"], "qwen2.5:14b")

        # Reset
        resetted = reset_config()
        self.assertEqual(resetted["remote_url"], "")
        self.assertEqual(resetted["auth_token"], "")

if __name__ == "__main__":
    unittest.main()
