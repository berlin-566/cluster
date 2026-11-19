import unittest
from cluster.fetch import search_models, calculate_vram_fit, format_search_table, is_model_available, LM_STUDIO_CATALOG, OLLAMA_CATALOG, HF_CATALOG

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
        # Check providers include Ollama, LM Studio, Hugging Face
        providers = {r["provider"] for r in results}
        self.assertIn("Ollama", providers)
        self.assertIn("LM Studio", providers)
        self.assertIn("Hugging Face", providers)

    def test_search_models_lm_studio(self):
        results = search_models("lm studio", pooled_vram_gb=64.0)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["provider"], "LM Studio")
            self.assertTrue(r["pull_id"].startswith("hf.co/"))

    def test_is_model_available(self):
        # Local model names without slash always return True
        self.assertTrue(is_model_available("llama3.1:8b"))
        self.assertTrue(is_model_available("qwen2.5:14b"))

    def test_all_catalog_models_have_pull_id_and_provider(self):
        for item in LM_STUDIO_CATALOG:
            self.assertEqual(item["provider"], "LM Studio")
            self.assertTrue(item["pull_id"].startswith("hf.co/"))
        for item in OLLAMA_CATALOG:
            self.assertEqual(item["provider"], "Ollama")
            self.assertIn("pull_id", item)
        for item in HF_CATALOG:
            self.assertIn(item["provider"], ("Hugging Face", "HuggingChat / HF"))
            self.assertTrue(item["pull_id"].startswith("hf.co/"))

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
