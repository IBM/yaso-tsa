# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import pandas as pd

from yaso_tsa.infra.CategoricalLabel import CategoricalLabel
from yaso_tsa.infra.LabeledSpan import LabeledSpan
from yaso_tsa.infra.SentimentTargets import SentimentTargets


class LabeledCluster:

    def __init__(self, *, labeled_spans=[], labeled_clusters=[]):
        self.labeled_spans = []
        for item in labeled_spans:
            self.__append(item)
        for group in labeled_clusters:
            for item in group.labeled_spans:
                self.__append(item)
        self.span = self.__span_as_interval()
        LabeledCluster.assert_texts(self.labeled_spans)
        if len(self.labeled_spans) == 0:
            raise ValueError('Cannot create an empty object')
        self.text = self.labeled_spans[0].text

    def __repr__(self):
        return f"<LabeledCluster (size={len(self.labeled_spans)}, span={self.span}), " \
               f"LabeledSpans: {self.labeled_spans}>"

    def get_labeled_text(self):
        return self.text[self.span.left:self.span.right]

    def append(self, item: LabeledSpan):
        if item.text != self.text:
            raise ValueError(f'Item text "{item.text}" is different from group text "{self.text}"')
        self.__append(item)
        self.span = self.__span_as_interval()

    def __append(self, item: LabeledSpan):
        if item not in self.labeled_spans:
            self.labeled_spans.append(item)

    def __span_as_interval(self):
        begin = min(item.begin for item in self.labeled_spans)
        end = max(item.end for item in self.labeled_spans)
        return pd.Interval(begin, end, closed='both')

    def overlaps(self, labeled_span: LabeledSpan):
        return self.span.overlaps(labeled_span.span_as_interval()) and self.text == labeled_span.text

    def contains_exact(self, item_to_check: LabeledSpan):
        return any(labeled_span.is_same_span(item_to_check) for labeled_span in self.labeled_spans)

    def get_aggregated_label(self):
        return CategoricalLabel.add([item.label for item in self.labeled_spans])

    def is_consistent_label(self):
        return self.get_aggregated_label().is_unanimous()

    def majority_label(self):
        return self.get_aggregated_label().most_common_label

    @staticmethod
    def assert_texts(labeled_spans):
        assert len(set(item.text for item in labeled_spans)) == 1

    @staticmethod
    def create_clusters(labeled_spans):
        result = []
        LabeledCluster.assert_texts(labeled_spans)
        for labeled_span in labeled_spans:
            span = labeled_span.span_as_interval()
            overlapping_groups = [group for group in result
                                  if span.overlaps(group.span)]
            num_found_groups = len(overlapping_groups)
            logging.debug(f'Found {num_found_groups} groups')
            if num_found_groups == 0:
                result.append(LabeledCluster(labeled_spans=[labeled_span]))
            elif num_found_groups == 1:
                overlapping_groups[0].append(labeled_span)
            elif num_found_groups > 1:
                merged_group = LabeledCluster(
                    labeled_spans=[labeled_span],
                    labeled_clusters=overlapping_groups
                )
                result.append(merged_group)
                for overlapping_group in overlapping_groups:
                    result.remove(overlapping_group)
        return result

    @staticmethod
    def to_frame(target_groups, detail_members=True):
        group_dictionaries = []
        for group in target_groups:
            group_as_dictionary = {
                SentimentTargets.SENTENCE_TEXT: group.text,
                SentimentTargets.TARGET_TEXT: group.get_labeled_text(),
                SentimentTargets.TARGET_BEGIN: group.span.left,
                SentimentTargets.TARGET_END: group.span.right,
                SentimentTargets.TARGET_SENTIMENT: group.majority_label(),
                'num_items': len(group.labeled_spans),
                'items': [(labeled_span.get_labeled_text(), labeled_span.begin, labeled_span.end, labeled_span.label)
                          for labeled_span in group.labeled_spans]
            }
            from targeted_sentiment_analysis.ComprehensiveLabeling.ComprehensiveSentimentTargets import \
                ComprehensiveSentimentTargets
            detection_columns = ComprehensiveSentimentTargets.detection_columns(
                ['positive', 'negative', 'mixed'], as_dictionary=True)
            for sentiment_label, detection_column in detection_columns.items():
                group_as_dictionary[detection_column] = group.get_aggregated_label().counter[sentiment_label]
            if detail_members:
                [group_as_dictionary.update({
                    f'{SentimentTargets.TARGET_TEXT}_{span_i}': labeled_span.get_labeled_text(),
                    f'{SentimentTargets.TARGET_BEGIN}_{span_i}': labeled_span.begin,
                    f'{SentimentTargets.TARGET_END}_{span_i}': labeled_span.end,
                    f'{SentimentTargets.TARGET_SENTIMENT}_{span_i}': labeled_span.label
                }) for span_i, labeled_span in enumerate(group.labeled_spans)]
            group_dictionaries.append(group_as_dictionary)
        return pd.DataFrame(group_dictionaries)
