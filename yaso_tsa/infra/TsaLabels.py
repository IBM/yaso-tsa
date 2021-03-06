# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from typing import List

import numpy
import pandas as pd

from yaso_tsa.infra.LabeledCluster import LabeledCluster
from yaso_tsa.infra.LabeledTarget import LabeledTarget
from yaso_tsa.infra.SentimentTargets import SentimentTargets, SENTENCE_TEXT, TARGET_TEXT, TARGET_SENTIMENT, \
    TARGET_BEGIN, TARGET_END
from yaso_tsa.infra.TsaData import TsaData

TARGET_CONFIDENCE = 'confidence'
SENTIMENT_ANSWER_NUM_LABELERS = 'num_annotations'
POSITIVE_ANSWER_COUNT = 'sentiment_positive'
NEGATIVE_ANSWER_COUNT = 'sentiment_negative'
NONE_ANSWER_COUNT = 'sentiment_none'
MIXED_ANSWER_COUNT = 'sentiment_mixed'
DETECTION_POSITIVE = 'detected_positive'
DETECTION_NEGATIVE = 'detected_negative'
DETECTION_MIXED = 'detected_mixed'

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


class TsaLabels:

    LABEL_COLUMNS = [
        TARGET_SENTIMENT, TARGET_CONFIDENCE, SENTIMENT_ANSWER_NUM_LABELERS,
        MIXED_ANSWER_COUNT, NONE_ANSWER_COUNT, NEGATIVE_ANSWER_COUNT,
        POSITIVE_ANSWER_COUNT
    ]

    @staticmethod
    def read_json(path, meta_fields=[]):
        as_tsa_data = TsaData.read_json(path, meta_fields=meta_fields)
        sentiment_targets = as_tsa_data.get_sentiment_targets()
        sentiment_targets.get_frame().rename(
            columns=lambda x: x.replace('detected_by.', ''),
            inplace=True
        )
        return TsaLabels(
            frame=sentiment_targets.get_frame(),
            sentences=as_tsa_data.get_sentences_frame()
        )

    def __init__(self, *, frame=None, sentences=pd.DataFrame(columns=[SENTENCE_TEXT])):
        if frame is None:
            frame = pd.DataFrame(
                columns=SentimentTargets.KEY_COLUMNS + TsaLabels.LABEL_COLUMNS
            )
        self.frame = frame
        self.sentences = sentences

    def __repr__(self):
        return f"<TsaLabels " \
               f"labeled: {self.get_num_labels()}, sentences: {self.get_num_sentences()}>"

    def empty(self):
        return self.frame.empty and self.sentences.empty

    def get_frame(self):
        return self.frame

    def get_sentences_frame(self):
        return self.sentences

    def get_num_labels(self):
        return len(self.frame)

    def is_valid_target(self):
        return self.frame[TARGET_SENTIMENT] != 'none'

    def get_valid_targets(self):
        valid_targets = self.frame[self.is_valid_target()]
        return TsaLabels(frame=valid_targets, sentences=self.sentences)

    def get_non_targets(self):
        non_targets = self.frame[~self.is_valid_target()]
        return TsaLabels(frame=non_targets, sentences=self.sentences)

    def get_num_sentences(self):
        return len(self.get_sentences())

    def get_sentences(self):
        return list(self.sentences[SENTENCE_TEXT])

    def is_labeled(self, target_text, text):
        return ((self.frame[TARGET_TEXT] == target_text) &
                (self.frame[SENTENCE_TEXT] == text)).any()

    def as_labeled_spans(self):
        if self.frame.empty:
            return []
        else:
            return self.frame.apply(lambda x: LabeledTarget.create(row=x), axis=1)

    def as_labeled_clusters(self) -> List[LabeledCluster]:
        labeled_targets = self.frame.groupby(SENTENCE_TEXT).apply(
            lambda x: LabeledTarget.create(frame=x))
        result = labeled_targets.apply(LabeledCluster.create_clusters)
        result = result.values
        result = numpy.hstack(result) if len(result) > 0 else result
        return result

    def shuffle(self):
        """
        Shuffle the order of the labels and the sentences.
        :return: A shuffled TsaLabels object.
        """
        return TsaLabels(
            frame=self.frame.sample(frac=1),
            sentences=self.sentences.sample(frac=1)
        )

    def split_by_sentences(self, split_percentage):
        """
        Split the labes into two parts, given a desired size (in percentage)
         for one of the parts. The split is done according to the current order
         of the sentences.
        :param split_percentage: The desired size (in percentage) of the size for
        the first part. For example, 0.8 will create two parts, the first containing
        80% of the labels, and the second containing 20% of the data.
        :return: A list containing two TsaLabels objects, for the splitted parts.
        """
        sentences = self.get_sentences()
        last_index = int(len(sentences) * split_percentage)
        first_split_sentences = sentences[:last_index]
        second_split_sentences = sentences[last_index:]
        return [self.select_sentences(first_split_sentences),
                self.select_sentences(second_split_sentences)]

    def to_csv(self, path, shuffle=False):
        output = self.frame
        if shuffle:
            output = output.sample(frac=1)
        output.to_csv(path)

    def to_json(self, path, additional_fields=[], shuffle=False):

        def get_optional(single_target, column_name, default_value=0):
            if column_name in single_target.index:
                return single_target[column_name]
            else:
                return default_value

        def to_dict(label):
            result = {
                TARGET_CONFIDENCE: get_optional(label, TARGET_CONFIDENCE),
                SENTIMENT_ANSWER_NUM_LABELERS: get_optional(label, SENTIMENT_ANSWER_NUM_LABELERS),
                POSITIVE_ANSWER_COUNT: get_optional(label, POSITIVE_ANSWER_COUNT),
                NEGATIVE_ANSWER_COUNT: get_optional(label, NEGATIVE_ANSWER_COUNT),
                MIXED_ANSWER_COUNT: get_optional(label, MIXED_ANSWER_COUNT),
                NONE_ANSWER_COUNT: get_optional(label, NONE_ANSWER_COUNT),
                "detected_by": {
                    DETECTION_POSITIVE: get_optional(label, DETECTION_POSITIVE),
                    DETECTION_NEGATIVE: get_optional(label, DETECTION_NEGATIVE),
                    DETECTION_MIXED: get_optional(label, DETECTION_MIXED)
                }
            }
            result.update({field: get_optional(label, field) for field in additional_fields})
            return result

        from yaso_tsa.infra.TsaData import TsaData
        as_tsa_data = TsaData(SentimentTargets(frame=self.frame), sentences=self.sentences)
        as_tsa_data.to_json(path, to_dict=to_dict, shuffle=shuffle)

    def add_detection_annotations(self, detection_annotations):
        self.frame = self.frame.merge(
            right=detection_annotations,
            on=SentimentTargets.KEY_COLUMNS
        )

    def get_answer_counts(self, normalize=False):
        result = self.frame[TARGET_SENTIMENT].value_counts(normalize=normalize)

        if normalize:
            result = result.rename(lambda x: f'% {x}')
        return result

    def add_property(self, property_name, property_value):
        self.frame[property_name] = property_value

    def add_sentence_property(self, property_name, property_value):
        self.sentences[property_name] = property_value

    def append(self, other):
        return TsaLabels(
            frame=self.frame.append(other.frame, ignore_index=True),
            sentences=self.sentences.append(other.sentences, ignore_index=True)
        )

    def add_confidence_bin(self, bins=[0, 0.7, 0.8, 0.9, 1.1]):
        self.frame['confidence_bin'] = pd.cut(
            self.frame[TARGET_CONFIDENCE].copy(),
            include_lowest=True,
            right=False,
            bins=bins
        )

    def sample_by_confidence(self, num_to_sample):
        sampled_frame = self.frame.groupby('confidence_bin').apply(
            lambda d: d.sample(n=num_to_sample)).reset_index(drop=True)
        return TsaLabels(frame=sampled_frame)

    def get_confidence_counts(self, bins=[0, 0.7, 0.8, 0.9, 1.1], normalize=False):
        result = pd.cut(
            self.frame[TARGET_CONFIDENCE],
            include_lowest=True,
            right=False,
            bins=bins
        ).value_counts(normalize=normalize)
        if normalize:
            result = result.rename(lambda x: f'% {x}')
        return result

    def select_sentences(self, sentences):
        return TsaLabels(
            frame=self.frame[self.frame[SENTENCE_TEXT].isin(sentences)],
            sentences=self.sentences[self.sentences[SENTENCE_TEXT].isin(sentences)]
        )

    def get_targets_with_confidence(self, confidence_condition):
        selected = self.frame[confidence_condition(self.frame[TARGET_CONFIDENCE])]
        return TsaLabels(
            frame=selected,
            sentences=self.get_sentences_frame()
        )

    def get_high_confidence_labels(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        return self.get_targets_with_confidence(lambda confidence: confidence >= confidence_threshold)

    def get_low_confidence_labels(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        return self.get_targets_with_confidence(lambda confidence: confidence < confidence_threshold)

    def remove_targets_from_domain(self, domain):
        self.frame = self.frame[self.frame['domain'] != domain]

    def remove(self, other):
        from labeling.collected_utils import remove_rows
        result = remove_rows(
            source=self.frame,
            items_to_remove=other.frame,
            key_columns=SentimentTargets.KEY_COLUMNS)
        return TsaLabels(frame=result)

    def as_sentiment_targets(self, extra_columns=[]):
        returned_columns = SentimentTargets.KEY_COLUMNS + [TARGET_SENTIMENT] + extra_columns
        sentiment_targets_frame = self.frame[returned_columns].copy()
        return SentimentTargets(frame=sentiment_targets_frame)

    def extend_labels(self):
        result = self.extend_labels_by_removing_prefix(prefix='the ')
        result = result.extend_lables_by_including_prefix(prefix='the ')
        return result

    def extend_labels_by_removing_prefix(self, prefix):
        '''
        Project all labels starting with "<prefix> <x>", to "<x>"
        For example in "The car is nice", for the prefix "The ", if "The car" is labeled positive, then
        this operation will add the label "car" with the same sentiment.
        :return:
            A TsaLabels() object containing the union of the original labels and the extended labels.
        '''
        target_starts_with_prefix = self.frame[TARGET_TEXT].str.lower().str.startswith(prefix)
        return self._add_new_offseted_labels(target_starts_with_prefix, offset=len(prefix))

    def extend_lables_by_including_prefix(self, prefix):
        '''
        Project all labels starting with "<x>" to "<prefix> <x>", if the prefix indeed appears
        before <x> in the sentence.
        For example in "The car is nice", for the prefix "The ", if "car" is labeled positive, then
        this operation will add the label "The car" with the same sentiment.
        :return:
            A TsaLabels() object containing the union of the original labels and the extended labels.
        '''

        def has_prefix(row):
            prefix_end = row[TARGET_BEGIN]
            prefix_begin = prefix_end - len(prefix)
            if prefix_begin >=0:
                target_prefix = row[SENTENCE_TEXT][prefix_begin:prefix_end]
                if target_prefix.lower() == prefix.lower():
                    return True
            return False

        with_prefix = self.frame.apply(has_prefix, axis=1)
        return self._add_new_offseted_labels(with_prefix, offset=-len(prefix))

    def _add_new_offseted_labels(self, labels_to_add, offset):
        new_labels = self.frame[labels_to_add].copy()
        new_labels[TARGET_BEGIN] = new_labels[TARGET_BEGIN].apply(lambda x: x + offset)
        new_labels[TARGET_TEXT] = new_labels.apply(
            lambda row: row[SENTENCE_TEXT][row[TARGET_BEGIN]:row[TARGET_END]], axis=1, result_type='reduce')
        extended_frame = pd.concat([self.frame, new_labels], ignore_index=True)
        # keep the first duplicate, which is an the original label
        # So, for example, if both "The <X>" and <X>" are originally labeled,
        # the extension of the label from "The <X>" to <X> is discarded, and the original label for <X> is kept.
        extended_without_duplicates = extended_frame.drop_duplicates(subset=SentimentTargets.KEY_COLUMNS, keep='first')
        return TsaLabels(frame=extended_without_duplicates, sentences=self.sentences.copy())


def get_available_sentiment_values():
    available_sentiment_values = ['mixed', 'none', 'negative', 'positive']
    return available_sentiment_values


