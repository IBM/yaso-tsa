# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import unittest
import pandas as pd

from yaso_tsa.infra.TsaData import TsaData
from yaso_tsa.tests.test_utils import get_test_data_path, get_test_data_path_2, get_test_xml_data_path


class TestTsaData(unittest.TestCase):

    def test_read_json(self):
        tsa_data = TsaData.read_json(path=get_test_data_path())
        self.assertEqual(len(tsa_data.get_sentences()), 3)

    def test_read_json(self):
        tsa_data = TsaData.read_jsons(files=[
            get_test_data_path(), get_test_data_path_2()
        ])
        self.assertEqual(len(tsa_data.get_sentences()), 5)
        tsa_data = TsaData.read_jsons(files=[
                get_test_data_path(), get_test_data_path_2()
            ],
            verbose=True
        )
        self.assertEqual(len(tsa_data.get_sentences()), 5)

    def test_read_xml(self):
        tsa_data = TsaData.read_xml(path=get_test_xml_data_path())
        self.assertEqual(len(tsa_data.get_sentences()), 3)

    def test_write_read(self):
        tsa_data = TsaData.read_json(path=get_test_data_path())
        test_file_name = 'TsaData.json'
        tsa_data.to_json(test_file_name)
        loaded = TsaData.read_json(path=test_file_name)
        self.assertListEqual(tsa_data.get_sentences(), loaded.get_sentences())
        pd.testing.assert_frame_equal(
            tsa_data.get_sentiment_targets().get_frame(),
            loaded.get_sentiment_targets().get_frame()
        )
        os.remove(test_file_name)

    def test_get_name(self):
        tsa_data = TsaData.read_json(path=get_test_data_path())
        self.assertEqual(tsa_data.get_name(), 'data\\test_data.json')

    def test_get_sentences_with_predictions(self):
        tsa_data = TsaData.read_json(path=get_test_data_path())
        self.assertEqual(len(tsa_data.get_sentences_with_predictions()), 2)

    def test_get_sentences_without_predictions(self):
        tsa_data = TsaData.read_json(path=get_test_data_path())
        self.assertEqual(len(tsa_data.get_sentences_without_targets()), 1)


