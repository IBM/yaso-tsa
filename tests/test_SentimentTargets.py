# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest
import os
import pandas as pd

from yaso_tsa.infra.SentimentTargets import SentimentTargets, SENTENCE_TEXT, TARGET_TEXT, TARGET_BEGIN, TARGET_END
from test_utils import get_test_data_path


class TestSentimentTargets(unittest.TestCase):

    def test_from_json(self):
        sentiment_targets = SentimentTargets.read_json(path=get_test_data_path())
        self.assertEqual(sentiment_targets.get_num_targets(), 3)
        # Only sentences with targets are loaded, so the expected number of sentences is 2
        sentences = sentiment_targets.get_sentences()
        self.assertEqual(len(sentences), 2)
        self.assertIn("This is a great car", sentences)
        self.assertIn("This is a great car, with an ugly color", sentences)

    def test_write_read_sentences(self):
        sentiment_targets = SentimentTargets.read_json(path=get_test_data_path())
        test_file_name = 'sentences.csv'
        sentiment_targets.write_sentences(path=test_file_name)
        sentences = pd.read_csv(test_file_name)
        self.assertEqual(len(sentences), 2)
        sentences = sentences['0']
        self.assertEqual(sentences.iloc[0], 'This is a great car')
        self.assertEqual(sentences.iloc[1], 'This is a great car, with an ugly color')
        os.remove(test_file_name)

    def test_write_read_to_csv(self):
        sentiment_targets = SentimentTargets.read_json(path=get_test_data_path())
        test_file_name = 'targets.csv'
        sentiment_targets.to_csv(test_file_name)
        read_targets = pd.read_csv(test_file_name)
        self.assertEqual(len(read_targets), 3)
        self.assertEqual(len(read_targets.columns), 4)
        self.assertIn(SENTENCE_TEXT, read_targets.columns)
        self.assertIn(TARGET_TEXT, read_targets.columns)
        self.assertIn(TARGET_BEGIN, read_targets.columns)
        self.assertIn(TARGET_END, read_targets.columns)
        os.remove(test_file_name)

    def test_select_targets(self):
        sentiment_targets = SentimentTargets.read_json(path=get_test_data_path())
        selected = sentiment_targets.select_targets(required_sentiment=['positive'])
        self.assertEqual(selected.get_num_targets(), 3)
        selected = sentiment_targets.select_targets(required_sentiment=['negative'])
        self.assertEqual(selected.get_num_targets(), 0)
