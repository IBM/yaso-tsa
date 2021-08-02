# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
import os

import pandas as pd

from yaso_tsa.infra.TsaLabels import TsaLabels
from yaso_tsa.tests.test_utils import get_test_labels_path


class TestTsaData(unittest.TestCase):

    def test_read_json(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        self.assertEqual(tsa_labels.get_num_labels(), 4)
        self.assertEqual(len(tsa_labels.get_sentences()), 3)

    def test_empty_object(self):
        tsa_labels = TsaLabels()
        self.assertEqual(tsa_labels.get_num_labels(), 0)
        self.assertEqual(len(tsa_labels.get_sentences()), 0)

    def test_read_write_json(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        test_file_name = 'TsaLabels.json'
        tsa_labels.to_json(test_file_name)
        loaded = TsaLabels.read_json(test_file_name)
        pd.testing.assert_frame_equal(tsa_labels.get_frame(), loaded.get_frame())
        os.remove(test_file_name)

    def test_get_valid_targets(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        valid_targets = tsa_labels.get_valid_targets()
        self.assertEqual(valid_targets.get_num_labels(), 3)

    def test_get_non_targets(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        valid_targets = tsa_labels.get_non_targets()
        self.assertEqual(valid_targets.get_num_labels(), 1)

    def test_get_high_confidence_labels(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        high_confidence = tsa_labels.get_high_confidence_labels()
        self.assertEqual(high_confidence.get_num_labels(), 3)

    def test_get_low_confidence_labels(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        low_confidence = tsa_labels.get_low_confidence_labels()
        self.assertEqual(low_confidence.get_num_labels(), 1)

    def test_is_labeled(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        self.assertFalse(tsa_labels.is_labeled(target_text='non existent', text='sentence that is not in tsa_labels'))
        sentence = 'This is a great car, with an ugly color'
        self.assertTrue(tsa_labels.is_labeled(target_text='car', text=sentence))
        self.assertFalse(tsa_labels.is_labeled(target_text='an', text=sentence))

    def test_as_labeled_clusters(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        labeled_clusters = tsa_labels.as_labeled_clusters()
        self.assertEqual(len(labeled_clusters), 4)

