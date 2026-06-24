from __future__ import annotations

import unittest

from demo.run_ingest_demo import CONFIG_PATH, load_demo_config


class ConfigParserTest(unittest.TestCase):
    def test_loads_demo_defaults(self) -> None:
        config = load_demo_config(CONFIG_PATH)
        self.assertEqual(config["ingest"]["ocr_mode"], "auto")
        self.assertEqual(config["ingest"]["table_extractor"], "configured")
        self.assertTrue(config["demo"]["save_overlay"])


if __name__ == "__main__":
    unittest.main()

