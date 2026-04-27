import base64
import importlib
import json
import os
import tempfile
import unittest


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.snapshot_path = os.path.join(cls.tmp_dir.name, "snapshots-test.json")
        os.environ["SNAPSHOT_STORE_PATH"] = cls.snapshot_path

        cls.app_module = importlib.import_module("app")
        cls.app_module = importlib.reload(cls.app_module)
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_predict_returns_extended_fields(self):
        response = self.client.post(
            "/api/predict",
            json={"text": "Great product but terrible delivery", "model": "distilbert"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("label", payload)
        self.assertIn("confidence", payload)
        self.assertIn("explainability", payload)
        self.assertIn("uncertainty", payload)
        self.assertIn("language", payload)

    def test_compare_returns_all_models(self):
        response = self.client.post("/api/compare", json={"text": "This update is fine"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("results", payload)
        self.assertEqual(len(payload["results"]), 3)

    def test_export_report_returns_pdf_base64(self):
        response = self.client.post(
            "/api/export-report",
            json={
                "title": "Report",
                "distribution": {"positive": 3, "neutral": 2, "negative": 1},
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload.get("format"), "pdf-base64")
        raw = base64.b64decode(payload.get("content_base64", ""))
        self.assertTrue(raw.startswith(b"%PDF"))

    def test_snapshot_save_is_persisted_to_file(self):
        response = self.client.post(
            "/api/save-snapshot",
            json={"title": "Persist me", "role": "manager", "data": {"a": 1}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(os.path.exists(self.snapshot_path))

        with open(self.snapshot_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        self.assertTrue(any(item.get("title") == "Persist me" for item in data.values()))

    def test_simulate_scenario_changes_negative_ratio(self):
        response = self.client.post(
            "/api/simulate-scenario",
            json={
                "distribution": {"positive": 40, "neutral": 30, "negative": 30},
                "delta_negative_pct": 20,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertGreater(payload["after"]["negative"], payload["before"]["negative"])


if __name__ == "__main__":
    unittest.main()
