import asyncio
import os
import tempfile
import unittest

from cluster.tui import ClusterTUIApp, display_gpu, print_banner, copy_to_clipboard, HAS_TEXTUAL

class TestTUI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_env = os.environ.get("CLUSTER_CONFIG_DIR")
        os.environ["CLUSTER_CONFIG_DIR"] = self.tmpdir.name

    def tearDown(self):
        if self.orig_env is not None:
            os.environ["CLUSTER_CONFIG_DIR"] = self.orig_env
        else:
            os.environ.pop("CLUSTER_CONFIG_DIR", None)
        self.tmpdir.cleanup()

    def test_legacy_cli_helpers(self):
        gpu_info = {
            "gpu_name": "Tesla T4",
            "vram_used_mb": 6400,
            "vram_total_mb": 16000,
            "gpu_util_pct": 85,
            "temp_c": 48,
        }
        # Verify legacy helpers do not crash
        display_gpu(gpu_info)
        print_banner("qwen2.5:14b", gpu_info, 12.5)

    def test_copy_to_clipboard_fallback(self):
        res = copy_to_clipboard("test-command")
        self.assertIsInstance(res, bool)

    def test_textual_app_lifecycle_and_tabs(self):
        if not HAS_TEXTUAL:
            self.skipTest("Textual is not installed")
        async def run_async():
            app = ClusterTUIApp(hub_port=2424, proxy_port=8080)
            async with app.run_test(size=(140, 50)) as pilot:
                # 1. Verify initial state
                self.assertEqual(app.cluster_health, "HEALTHY")
                self.assertIn("64.0 GB", app.pooled_vram_str)

                # 2. Test DAG Node selections
                await pilot.click("#btn-node-admin")
                self.assertEqual(app.selected_node_id, "admin")

                await pilot.click("#btn-node-leader")
                self.assertEqual(app.selected_node_id, "leader")

                await pilot.click("#btn-node-worker1")
                self.assertEqual(app.selected_node_id, "worker1")

                await pilot.click("#btn-node-worker2")
                self.assertEqual(app.selected_node_id, "worker2")

                await pilot.click("#btn-node-worker3")
                self.assertEqual(app.selected_node_id, "worker3")

                btn_members = app.query_one("#btn-node-members")
                btn_members.scroll_visible()
                await pilot.pause()
                await pilot.click("#btn-node-members")
                self.assertEqual(app.selected_node_id, "members")

                # 3. Test Action buttons
                app.action_copy_reconnect()
                app.action_ping_rpc()
                app.action_evict_node()

                # 4. Test Tab Navigation
                app.action_switch_tab("tab-models")
                await pilot.pause()
                # Test deploy button
                await pilot.click("#model-deploy-btn")

                app.action_switch_tab("tab-chat")
                await pilot.pause()
                # Test sending chat input
                chat_input = app.query_one("#chat-input")
                chat_input.value = "Hello distributed mesh!"
                await pilot.click("#btn-chat-send")

                app.action_switch_tab("tab-team")
                await pilot.pause()
                # Verify team table has rows
                team_table = app.query_one("#team-table")
                self.assertGreater(team_table.row_count, 0)

                app.action_switch_tab("tab-logs")
                await pilot.pause()
                # Test adding log event
                await pilot.click("#btn-test-log")
                await pilot.click("#btn-clear-logs")

        asyncio.run(run_async())

if __name__ == "__main__":
    unittest.main()
