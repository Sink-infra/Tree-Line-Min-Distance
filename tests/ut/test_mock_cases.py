"""Check Mock declarations, generated tensors, and mathematical references."""

import unittest

from tests.common.checks import check_declarations, check_generated_data
from tests.common.cases import MOCK_CASES as CASES


class MockTests(unittest.TestCase):
    def test_declarations(self):
        check_declarations(self, CASES, "mock")

    def test_generated_data(self):
        check_generated_data(self, CASES, "mock")


if __name__ == "__main__":
    unittest.main()
