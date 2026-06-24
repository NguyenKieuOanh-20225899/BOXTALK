from __future__ import annotations

import unittest

from demo.adapters.output_adapter import page_slug


class OutputAdapterTest(unittest.TestCase):
    def test_page_slug_is_one_based_and_padded(self) -> None:
        self.assertEqual(page_slug(1), "page_01")
        self.assertEqual(page_slug(12), "page_12")


if __name__ == "__main__":
    unittest.main()

