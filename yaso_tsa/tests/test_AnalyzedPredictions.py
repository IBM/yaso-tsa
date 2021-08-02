# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from unittest import TestCase

from yaso_tsa.Analysis.AnalzyedPredictions import AnalyzedPredictions, EXACT_MATCHER, TARGET_EXTRACTION, F1, PRECISION, RECALL, \
    NUM_PREDICTIONS, NUM_CORRECT, TARGETED_SENTIMENT_ANALYSIS, SENTIMENT_CLASSIFICATION
from yaso_tsa.infra.TsaData import TsaData
from yaso_tsa.infra.TsaLabels import TsaLabels
from yaso_tsa.tests.test_utils import get_test_data_path, get_test_labels_path


class TestAnalyzedPredictions(TestCase):

    def create_analysis(self):
        predictions = TsaData.read_json(path=get_test_data_path())
        tsa_labels = TsaLabels.read_json(path=get_test_labels_path())
        return AnalyzedPredictions(
            tsa_data=predictions,
            labeled_data=tsa_labels,
            matchers=[EXACT_MATCHER]
        )

    def test_AnalyzedPredictions(self):
        analysis = self.create_analysis()
        self.assert_stats(
            analysis,
            TARGET_EXTRACTION,
            label=None,
            expected_num_predictions=3,
            expected_num_correct=3,
            expected_precision=1,
            expected_recall=1,
            expected_f1=1
        )

        self.assert_stats(
            analysis,
            SENTIMENT_CLASSIFICATION,
            label='positive',
            expected_num_predictions=3,
            expected_num_correct=2,
            expected_precision=2/3,
            expected_recall=1,
            expected_f1=0.8
        )

        self.assert_stats(
            analysis,
            SENTIMENT_CLASSIFICATION,
            label='negative',
            expected_num_predictions=0,
            expected_num_correct=0,
            expected_precision=0,
            expected_recall=0,
            expected_f1=0
        )

        self.assert_stats(
            analysis,
            TARGETED_SENTIMENT_ANALYSIS,
            label=None,
            expected_num_predictions=3,
            expected_num_correct=2,
            expected_precision=2/3,
            expected_recall=2/3,
            expected_f1=2/3
        )

    def assert_stats(self, analysis, task_name, label, expected_num_predictions,
                     expected_num_correct, expected_precision, expected_recall, expected_f1):

        def assert_stat(task_name, label, metric, expected_value):
            self.assertEqual(analysis.get_stat(
                task_name=task_name, label=label, metric=metric), expected_value)

        assert_stat(task_name, label, NUM_PREDICTIONS, expected_num_predictions)
        assert_stat(task_name, label, NUM_CORRECT, expected_num_correct)
        assert_stat(task_name, label, PRECISION, expected_precision)
        assert_stat(task_name, label, RECALL, expected_recall)
        assert_stat(task_name, label, F1, expected_f1)

    def test_get_stat(self):
        analysis = self.create_analysis()
        with self.assertRaises(RuntimeError):
            analysis.get_stat()
        with self.assertRaises(RuntimeError):
            analysis.get_stat(stat_name='stat name', task_name='task_name')
