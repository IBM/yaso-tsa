# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
import os

import pandas as pd

from yaso_tsa.infra.SentimentTargets import SENTENCE_TEXT, TARGET_END, TARGET_TEXT, TARGET_BEGIN
from yaso_tsa.infra.TsaLabels import TsaLabels, TARGET_CONFIDENCE
from yaso_tsa.tests.test_utils import get_test_labels_path, get_test_labels_with_the_path


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

    def test_extend_labels(self):
        tsa_labels = TsaLabels.read_json(path=get_test_labels_with_the_path())
        self.assertEqual(6, tsa_labels.get_num_labels())
        extended_labels = tsa_labels.extend_labels()
        self.assertEqual(10, extended_labels.get_num_labels())

        def assert_added(target, sentence):
            '''
                Verify that <target> was not originally present, and that its now added
            '''
            self.assertFalse(tsa_labels.is_labeled(target_text=target, text=sentence))
            self.assertTrue(extended_labels.is_labeled(target_text=target, text=sentence))

        # test with Cased 'The'
        assert_added('car', 'The car is nice.')
        assert_added('The picture', 'The picture is nice.')

        # test with lower case 'the'
        assert_added('car', 'I like the car.')
        assert_added('the picture', 'I like the picture.')

        # test existing labels aren't changed
        sentence = "This is a sentence without additions since the X and X are both labeled."
        self.assertTrue(tsa_labels.is_labeled(target_text='the X', text=sentence))
        self.assertTrue(tsa_labels.is_labeled(target_text='X', text=sentence))
        self.assertTrue(extended_labels.is_labeled(target_text='the X', text=sentence))
        self.assertTrue(extended_labels.is_labeled(target_text='X', text=sentence))

        extended_frame = extended_labels.get_frame()

        def get_confidence(target, sentence, begin, end):
            where = (
                (extended_frame[TARGET_TEXT] == target) &
                (extended_frame[SENTENCE_TEXT] == sentence) &
                (extended_frame[TARGET_BEGIN] == begin) &
                (extended_frame[TARGET_END] == end)
            )
            return extended_frame[where].iloc[0][TARGET_CONFIDENCE]

        # Check that the confidence in the labels remain as in the labels file
        # This makes sure that the labels are not wrongly duplicated with their confidence
        # and override the original label
        self.assertEqual(1, get_confidence(target='the X', sentence=sentence, begin=43, end=48))
        self.assertEqual(0.71, get_confidence(target='X', sentence=sentence, begin=47, end=48))
