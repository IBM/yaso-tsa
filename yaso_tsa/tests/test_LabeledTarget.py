# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import unittest

from yaso_tsa.infra.LabeledTarget import LabeledTarget
from yaso_tsa.infra.SentimentTargets import TARGET_SENTIMENT, TARGET_END, TARGET_BEGIN, SENTENCE_TEXT


class TestLabeledTarget(unittest.TestCase):

    def test_create(self):
        import pandas as pd
        data = [
            ['This is a nice sentence', 0, 4, 'none'],
            ['This is a nice sentence', 15, 23, 'positive']
        ]
        frame = pd.DataFrame(
            data=data,
            columns=[SENTENCE_TEXT, TARGET_BEGIN, TARGET_END, TARGET_SENTIMENT]
        )
        labeled_targets = LabeledTarget.create(frame=frame, index_label=TARGET_SENTIMENT)
        self.assertEqual(len(labeled_targets), 2)
