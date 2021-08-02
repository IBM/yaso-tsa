# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from unittest import TestCase
from collections import Counter

from yaso_tsa.infra.CategoricalLabel import CategoricalLabel


class TestCategoricalLabel(TestCase):
    def test_is_unanimous(self):
        categorical_label = CategoricalLabel(Counter(positive=5))
        self.assertTrue(categorical_label.is_unanimous())
        categorical_label = CategoricalLabel(Counter(positive=5, negative=2))
        self.assertFalse(categorical_label.is_unanimous())

    def test_is_inconclusive(self):
        categorical_label = CategoricalLabel(Counter(positive=5, negative=4))
        self.assertFalse(categorical_label.is_inconclusive)
        categorical_label = CategoricalLabel(Counter(positive=5, negative=5))
        self.assertTrue(categorical_label.is_inconclusive)
        categorical_label = CategoricalLabel(Counter(positive=5))
        self.assertFalse(categorical_label.is_inconclusive)

    def test_from_series_with_labels_to_columns(self):
        import pandas as pd
        labels = pd.Series({'positive_label': 2, 'negative_label': 1})
        label = CategoricalLabel.from_series(labels, labels_to_columns={
            'positive': 'positive_label',
            'negative': 'negative_label'
        })
        self.assertTrue(label.most_common_label, 'positive')
        self.assertTrue(label.most_common_count, 2)

    def test_from_series_with_label_column(self):
        import pandas as pd
        labels = pd.Series({'label': 'positive', 'other_column': 'some other data'})
        label = CategoricalLabel.from_series(labels, index_label='label')
        self.assertTrue(label.most_common_label, 'positive')
        self.assertTrue(label.most_common_count, 1)
