import unittest
from retry_policy import retry_delay

class RetryPolicyTests(unittest.TestCase):
    def test_declared_429_case(self):
        self.assertEqual(retry_delay(429, '3'), 3)

    def test_existing_5xx_contract(self):
        self.assertEqual(retry_delay(503, None), 2)

if __name__ == '__main__':
    unittest.main()
