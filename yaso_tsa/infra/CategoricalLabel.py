# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from collections import Counter
from typing import Dict


class CategoricalLabel:

    def __init__(self, counter):
        self.counter = counter
        top_two_common_answers = self.counter.most_common(2)
        most_common = top_two_common_answers[0]
        self.most_common_label, self.most_common_count = most_common[0], most_common[1]
        if len(top_two_common_answers) == 2:
            second_most_common = top_two_common_answers[1]
            second_most_common_count = second_most_common[1]
            self.is_inconclusive = self.most_common_count == second_most_common_count
        else:
            self.is_inconclusive = False

    def __repr__(self):
        return str(self.__dict__)

    def is_unanimous(self):
        return len(list(self.counter)) == 1

    @staticmethod
    def from_series(series, *, index_label=None, labels_to_columns: Dict[str, str] = None):
        counter = Counter()
        initialized = False
        if labels_to_columns:
            # try to load specific label counts from given column names
            for label, column_name in labels_to_columns.items():
                if column_name in series.index:
                    label_value = series[column_name]
                    if label_value > 0:
                        initialized = True
                        counter.update({label: label_value})
        if not initialized and index_label and index_label in series.index:
            # load from index_label
            label = series[index_label]
            counter.update({label: 1})
        return CategoricalLabel(counter)

    @staticmethod
    def add(categorical_labels):
        counter = sum([label.counter for label in categorical_labels], Counter())
        return CategoricalLabel(counter)


