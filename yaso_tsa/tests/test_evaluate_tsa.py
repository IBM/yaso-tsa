# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from yaso_tsa import evaluate_tsa
from yaso_tsa.evaluate_tsa import PREDICTIONS_PATH, LABELS_PATH
from yaso_tsa.tests.test_utils import get_test_data_path, get_test_labels_path


class MyTestCase(unittest.TestCase):

    def test_no_arguments(self):
        with self.assertRaises(SystemExit) as cm:
            evaluate_tsa.main()

        self.assertEqual(cm.exception.code, 2)

    def test_with_correct_arguments(self):
        import sys as _sys
        _sys.argv.extend([
            PREDICTIONS_PATH, get_test_data_path(),
            LABELS_PATH, get_test_labels_path()
        ])
        evaluate_tsa.main()


if __name__ == '__main__':
    unittest.main()
