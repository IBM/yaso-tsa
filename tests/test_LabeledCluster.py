# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from unittest import TestCase

from yaso_tsa.infra.LabeledCluster import LabeledCluster
from yaso_tsa.infra.LabeledSpan import LabeledSpan
import pandas as pd


class TestLabeledSpan(TestCase):

    def test_create_clusters(self):
        text = 'some text which is not very short'
        s = 'positive'
        labeled_targets = [
            LabeledSpan(text=text, begin=5, end=8, label=s),
            LabeledSpan(text=text, begin=5, end=15, label=s)
        ]

        clusters = LabeledCluster.create_clusters(labeled_targets)
        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(len(cluster.labeled_spans), 2)
        self.assertEqual(cluster.span, pd.Interval(5, 15, closed='both'))
        self.assertEqual(cluster.get_labeled_text(), 'text which')

        labeled_targets += [
            # cluster of size 2
            LabeledSpan(text=text, begin=0, end=2, label=s),
            LabeledSpan(text=text, begin=2, end=3, label=s),
            # additions to upper cluster
            # first an unrelated span, followed by a large span that should
            # "merge" this span (22:27) with the first cluster above
            LabeledSpan(text=text, begin=22, end=27, label=s),
            LabeledSpan(text=text, begin=5, end=27, label=s),
            # cluster of size 1
            LabeledSpan(text=text, begin=29, end=32, label=s)
        ]
        clusters = LabeledCluster.create_clusters(labeled_targets)
        self.assertEqual(len(clusters), 3)


