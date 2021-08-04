import unittest

from yaso_tsa import evaluate_tsa


class MyTestCase(unittest.TestCase):

    def test_no_arguments(self):
        with self.assertRaises(SystemExit) as cm:
            evaluate_tsa.main()

        self.assertEqual(cm.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
