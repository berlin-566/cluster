import unittest
from cluster.fetch import search_models, calculate_vram_fit, format_search_table

class TestFetch(unittest.TestCase):
    def test_search_models_qwen(self):
        results = search_models("qwen", pooled_vram_gb=64.0)
        self.assertGreater(len(results), 0)
        names = [r["name"].lower() for r in results]
        self.assertTrue(any("qwen" in n for n in names))
        # Check fit status is present
        self.assertIn("fit_status", results[0])
        self.assertIn(results[0]["fit_status"], ("FIT", "TIGHT", "EXCEEDS"))

    def test_search_models_empty_query(self):
        results = search_models("", pooled_vram_gb=32.0)
        self.assertGreater(len(results), 0)

    def test_calculate_vram_fit(self):
        # 10 GB required on 64 GB pooled -> FIT
        status, label = calculate_vram_fit(10.0, 64.0)
        self.assertEqual(status, "FIT")
        self.assertIn("Fits", label)

        # 60 GB required on 64 GB pooled -> TIGHT
        status, label = calculate_vram_fit(60.0, 64.0)
        self.assertEqual(status, "TIGHT")
        self.assertIn("Tight Fit", label)

        # 80 GB required on 64 GB pooled -> EXCEEDS
        status, label = calculate_vram_fit(80.0, 64.0)
        self.assertEqual(status, "EXCEEDS")
        self.assertIn("Exceeds", label)

    def test_format_search_table(self):
        results = [
            {
                "name": "qwen2.5:14b",
                "provider": "Ollama",
                "quant": "Q4_K_M",
                "size_gb": 9.0,
                "fit_status": "FIT",
                "fit_label": "Fits (10.5 GB / 64.0 GB)",
            }
        ]
        table = format_search_table(results)
        self.assertIn("MODEL", table)
        self.assertIn("qwen2.5:14b", table)
        self.assertIn("Ollama", table)
        self.assertIn("[FIT]", table)

if __name__ == "__main__":
    unittest.main()
