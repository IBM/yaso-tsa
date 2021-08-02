# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from yaso_tsa.infra.CategoricalLabel import CategoricalLabel
from yaso_tsa.infra.LabeledSpan import LabeledSpan
from yaso_tsa.infra.SentimentTargets import SENTENCE_TEXT, TARGET_BEGIN, TARGET_END, TARGET_SENTIMENT

SENTIMENT_LABEL_TYPE = 'sentiment'
DETECTION_LABEL_TYPE = 'detected'


class LabeledTarget:

    @staticmethod
    def create(*, frame=None, row=None, index_label=None, label_type=SENTIMENT_LABEL_TYPE):
        if row is not None:
            return LabeledTarget.__create_from_row(row, index_label, label_type)
        if frame is not None:
            return [LabeledTarget.__create_from_row(row, index_label, label_type) for _, row in frame.iterrows()]
        return None

    @staticmethod
    def __create_from_row(row, index_label=None, label_type=SENTIMENT_LABEL_TYPE):
        if index_label is None:
            index_label = TARGET_SENTIMENT
        return LabeledSpan(
            text=row[SENTENCE_TEXT],
            begin=int(row[TARGET_BEGIN]),
            end=int(row[TARGET_END]),
            label=CategoricalLabel.from_series(
                series=row,
                index_label=index_label,
                labels_to_columns={
                    label: f'{label_type}_{label}' for label in ['positive', 'negative', 'mixed']
                }
            )
        )
