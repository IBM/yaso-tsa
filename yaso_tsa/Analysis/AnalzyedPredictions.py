# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Callable, Tuple

import numpy
import pandas

from yaso_tsa.infra.LabeledCluster import LabeledCluster
from yaso_tsa.infra.LabeledSpan import LabeledSpan
from yaso_tsa.infra.SentimentTargets import SentimentTargets, TARGET_SCORE
from yaso_tsa.infra.TsaData import TsaData
from yaso_tsa.infra.TsaLabels import TsaLabels

IS_IGNORE_LABEL = 'is_ignore_label'

NUM_TARGET_PREDICTIONS = 'num predictions'

NUM_PREDICTION_SENTENCES = 'num prediction sentences'

NUM_LABELED_INPUT_SENTENCES = 'num labeled sentences'
NUM_INPUT_SENTENCES = 'num input sentences'
NUM_LABELED_INPUT_SENTENCES_WITH_PREDICTIONS = 'num labeled input sentences with predictions'
NUM_LABELS = 'num labels'
NUM_LABELED_CLUSTERS = 'num labeled clusters'

PERCENTAGE_COVERED_VALID_TARGET_GROUPS = '% covered valid target groups'
NUM_COVERED_VALID_TARGET_GROUPS = '# covered valid target groups'

MAJORITY_LABEL = 'label.majority_label'
LABELS = 'labels'
PREDICTIONS = 'predictions'
MATCH_TYPES = 'match_types'
PREDICTION = 'prediction'

# Evaluated tasks:
TARGET_EXTRACTION = 'target extraction'
SENTIMENT_CLASSIFICATION = 'sentiment prediction'
TARGETED_SENTIMENT_ANALYSIS = 'full pipeline'

# Evaluation metrics:
RECALL = 'recall'
PRECISION = 'precision'
F1 = 'F1'
F05 = 'F05'


# Evaluation counts:
IS_CORRECT = 'is correct'
NUM_CORRECT = 'num correct'
NUM_PREDICTIONS = 'num predictions'


def get_measure_name(task_name, *, label=None, metric=None):
    result = task_name
    if label:
        result = f'{result} - {label}'
    if metric:
        result = f'{result}: {metric}'
    return result


def compute_f05(precision, recall):
    return 5 * precision * recall / (precision + 4 * recall) if (precision + recall) > 0 else 0.


EXACT_MATCHER = ('exact', LabeledCluster.contains_exact)
OVERLAP_MATCHER = ('overlap', LabeledCluster.overlaps)


class AnalyzedPredictions:

    # Names of stats that are returned in the results stats
    NUM_UNLABELED = 'num_unlabeled'
    SENTIMENT_PREDICTION_MACRO_F1 = 'sentiment prediction: Macro-F1'
    NUM_NONE_PREDICTIONS_OF_VALID_TARGETS = 'num_none_predictions_with_valid_targets'

    # Names of columns the are included in the matched frames
    TARGET_EXTRACTION_CORRECT = get_measure_name(task_name=TARGET_EXTRACTION, metric=IS_CORRECT)
    SENTIMENT_PREDICTION_CORRECT = get_measure_name(task_name=SENTIMENT_CLASSIFICATION, metric=IS_CORRECT)
    FULL_PIPELINE_CORRECT = get_measure_name(task_name=TARGETED_SENTIMENT_ANALYSIS, metric=IS_CORRECT)

    def __init__(
        self,
        tsa_data: TsaData,
        labeled_data: TsaLabels,
        ignore_unlabeled=False,
        name=None,
        matchers=[EXACT_MATCHER],
        ignore_labels=TsaLabels()
    ):
        # restrict the labeled data to input sentences
        labeled_data = labeled_data.select_sentences(sentences=tsa_data.get_sentences())
        all_predictions = tsa_data.get_sentiment_targets()
        predictions = all_predictions.select_targets(required_sentiment=['positive', 'negative', 'mixed'])
        # restrict the evaluated predictions to labeled sentences
        predictions = predictions.select_sentences(labeled_data.get_sentences())
        valid_targets = labeled_data.get_valid_targets()
        non_targets = labeled_data.get_non_targets()
        labeled_clusters = valid_targets.as_labeled_clusters()
        self.stats = {
            NUM_INPUT_SENTENCES: len(tsa_data.get_sentences()),
            NUM_LABELED_INPUT_SENTENCES: labeled_data.get_num_sentences(),
            NUM_LABELED_INPUT_SENTENCES_WITH_PREDICTIONS: predictions.get_num_sentences(),
            NUM_TARGET_PREDICTIONS: predictions.get_num_targets(),
            'num labels': labeled_data.get_num_labels(),
            'num valid labels': valid_targets.get_num_labels(),
            NUM_LABELED_CLUSTERS: len(labeled_clusters),
            'num non-valid labels': non_targets.get_num_labels()}
        self.matched_predictions = self.match_predictions_to_labels(
            cluster_labels=labeled_clusters,
            predictions=predictions,
            matchers=matchers)
        self.matched_labels = self.match_labels_to_predictions(
            cluster_labels=labeled_clusters,
            predictions=all_predictions,
            matchers=matchers)
        self.matched_labels['is_covered_label'] = self.matched_labels['# predictions'] > 0
        self.stats[NUM_COVERED_VALID_TARGET_GROUPS] = self.matched_labels['is_covered_label'].sum()
        self.stats[PERCENTAGE_COVERED_VALID_TARGET_GROUPS] = self.stats[NUM_COVERED_VALID_TARGET_GROUPS] / \
                                                             self.stats[NUM_LABELED_CLUSTERS]

        def has_prediction(predictions, label):
            return any(prediction.label == label for prediction in predictions)
        self.matched_labels['is_matched_to_none_prediction'] = self.matched_labels.apply(
            lambda match: has_prediction(match[PREDICTIONS], 'none'), axis=1)
        self.stats[self.NUM_NONE_PREDICTIONS_OF_VALID_TARGETS] = \
            sum(self.matched_labels['is_matched_to_none_prediction'])

        self.match_to_non_targets(non_targets)
        self.match_to_ignore_labels(ignore_labels)
        self.calculate_correct_predictions()
        self.calculate_sentiment_correct_per_class()

        num_unlabeled = sum(self.matched_predictions['is_unlabeled'])
        self.stats['num_unlabeled'] = num_unlabeled
        num_predictions = len(self.matched_predictions)
        if ignore_unlabeled:
            num_predictions -= num_unlabeled
        num_ignore_labels = sum(self.matched_predictions[IS_IGNORE_LABEL]) \
            if 'is_ignore_label' in self.matched_predictions.columns else 0
        self.stats['num_ignore_labels'] = num_ignore_labels
        num_predictions -= num_ignore_labels
        target_extraction_correct = sum(self.matched_predictions[self.TARGET_EXTRACTION_CORRECT])
        self.calculate_precision_recall_f1(
            num_correctly_predicted=target_extraction_correct,
            num_predictions=num_predictions,
            num_valid_targets=len(labeled_clusters),
            task_name=TARGET_EXTRACTION)

        self.calculate_accuracy(
            num_correctly_predicted=sum(self.matched_predictions[self.SENTIMENT_PREDICTION_CORRECT].fillna(False)),
            num_predictions=target_extraction_correct,
            task_name=SENTIMENT_CLASSIFICATION)

        self.calculate_sentiment_marco_f1()

        self.calculate_precision_recall_f1(
            num_correctly_predicted=sum(self.matched_predictions[self.FULL_PIPELINE_CORRECT]),
            num_predictions=num_predictions,
            num_valid_targets=len(labeled_clusters),
            task_name=TARGETED_SENTIMENT_ANALYSIS)

        labeled_predictions = predictions.get_frame().merge(
            right=labeled_data.get_frame(),
            how='left',
            on=SentimentTargets.KEY_COLUMNS,
            validate='many_to_one',
            suffixes=['_predicted', '_label'])

        self.predictions_with_labels_frame = labeled_predictions
        self.name = name

        is_labeled = ~self.predictions_with_labels_frame['sentiment_label'].isna()
        self.labeled = self.predictions_with_labels_frame[is_labeled]

    @staticmethod
    def get_exact_matches(match, non_targets):
        prediction = match[PREDICTION]
        return [] if match['# labels'] > 0 else \
            [non_target for non_target in non_targets if non_target.is_same_span(prediction)]

    def match_to_non_targets(self, non_targets):
        matched_predictions = self.matched_predictions

        non_targets = non_targets.as_labeled_spans()

        matched_predictions['non targets'] = matched_predictions.apply(
            lambda match: AnalyzedPredictions.get_exact_matches(match, non_targets), axis=1)

        matched_predictions['is_labeled_non_target'] = matched_predictions.apply(
            lambda match: None if match['# labels'] > 0 else len(match['non targets']) > 0, axis=1)
        matched_predictions['is_unlabeled'] = matched_predictions.apply(
            lambda match: False if match['# labels'] > 0 else len(match['non targets']) == 0, axis=1)

    def match_to_ignore_labels(self, ignore_labels: TsaLabels):
        matched_predictions = self.matched_predictions
        if ignore_labels.get_num_labels() > 0:
            ignore_labels_spans = ignore_labels.as_labeled_spans()

            matched_predictions['ignore labels'] = matched_predictions.apply(
                lambda match: AnalyzedPredictions.get_exact_matches(match, ignore_labels_spans), axis=1)
            matched_predictions[IS_IGNORE_LABEL] = matched_predictions.apply(
                lambda match: False if match['# labels'] > 0 else len(match['ignore labels']) > 0, axis=1)
        else:
            matched_predictions['ignore labels'] = False
            matched_predictions[IS_IGNORE_LABEL] = False

    @staticmethod
    def correct_sentiment_column(label):
        return f'{AnalyzedPredictions.SENTIMENT_PREDICTION_CORRECT}: {label}'

    @staticmethod
    def get_available_labels(matched_predictions):
        return pandas.unique(matched_predictions[MAJORITY_LABEL].dropna())

    def calculate_sentiment_correct_per_class(self):
        matched_predictions = self.matched_predictions
        available_labels = AnalyzedPredictions.get_available_labels(matched_predictions)
        for available_label in available_labels:
            matched_predictions[AnalyzedPredictions.correct_sentiment_column(available_label)] = \
                matched_predictions.apply(
                    lambda match:
                        match[AnalyzedPredictions.SENTIMENT_PREDICTION_CORRECT] and
                        match[PREDICTION].label.most_common_label == available_label,
                    axis=1
                )

    @staticmethod
    def is_sentiment_correct(match):
        prediction: LabeledSpan = match[PREDICTION]
        label_clusters: List[LabeledCluster] = match['labels']
        label = None
        for labeled_cluster in label_clusters:
            for labeled_span in labeled_cluster.labeled_spans:
                if labeled_span.is_same_span(prediction):
                    if not label:
                        label = labeled_span.label.most_common_label
                    else:
                        raise ValueError(f"Already found a label for prediction {prediction}: {label}. New label: {labeled_span}")
        if not label:
            label = match[MAJORITY_LABEL]
        return prediction.label.most_common_label == label

    @staticmethod
    def get_majority_label(labels):
        if not labels:
            return None
        labeled_sentiments = [label.majority_label() for label in labels]
        value, count = Counter(labeled_sentiments).most_common()[0]
        majority_label = value
        return majority_label

    def calculate_precision_recall_f1(self, *, num_correctly_predicted, num_predictions, num_valid_targets, task_name):
        precision = (num_correctly_predicted / num_predictions) if num_predictions else 0
        recall = (num_correctly_predicted / num_valid_targets) if num_valid_targets else 0
        self.stats[get_measure_name(task_name, metric=NUM_CORRECT)] = num_correctly_predicted
        self.stats[get_measure_name(task_name, metric=NUM_PREDICTIONS)] = num_predictions
        self.stats[get_measure_name(task_name, metric=NUM_LABELS)] = num_valid_targets
        self.stats[get_measure_name(task_name, metric=PRECISION)] = precision
        self.stats[get_measure_name(task_name, metric=RECALL)] = recall
        f1 = statistics.harmonic_mean([precision, recall])
        self.stats[get_measure_name(task_name, metric=F1)] = f1
        f05 = compute_f05(precision, recall)

        self.stats[get_measure_name(task_name, metric=F05)] = f05
        return precision, recall, f1

    def calculate_accuracy(self, *, num_correctly_predicted, num_predictions, task_name):
        if num_predictions:
            accuracy = num_correctly_predicted / num_predictions
        else:
            accuracy = None
        self.stats[get_measure_name(task_name, metric=NUM_CORRECT)] = num_correctly_predicted
        self.stats[get_measure_name(task_name, metric=NUM_PREDICTIONS)] = num_predictions
        self.stats[get_measure_name(task_name, metric='accuracy')] = accuracy

    def calculate_sentiment_marco_f1(self):
        matched_predictions = self.matched_predictions
        available_labels = AnalyzedPredictions.get_available_labels(matched_predictions)
        available_labels = available_labels[available_labels != 'mixed']
        f1s = []
        task_name = 'sentiment prediction'
        for label in available_labels:
            _, _, f1 = self.calculate_precision_recall_f1(
                num_correctly_predicted=sum(matched_predictions[AnalyzedPredictions.correct_sentiment_column(label)]),
                num_predictions=sum(
                    matched_predictions[self.TARGET_EXTRACTION_CORRECT] &
                    (matched_predictions['prediction.sentiment'].apply(lambda x: x.most_common_label) == label)),
                num_valid_targets=sum(matched_predictions[self.TARGET_EXTRACTION_CORRECT] &
                                      (matched_predictions['label.majority_label'] == label)),
                task_name=get_measure_name(task_name, label=label))
            f1s.append(f1)
        average_f1 = numpy.mean(f1s)
        self.stats[f'{task_name}: Macro-F1'] = average_f1

    def calculate_correct_predictions(self):
        '''
        correct predictions are:
        target extraction: for each span if matched to a valid target then correct
        sentiment prediction: for each prediction with correctly extracted target, is sentiment correct
        full pipeline: target extraction and sentiment prediction are correct
        '''
        matched_predictions = self.matched_predictions
        matched_predictions[AnalyzedPredictions.TARGET_EXTRACTION_CORRECT] = \
            matched_predictions.apply(lambda match: len(match['labels']) > 0, axis=1)
        matched_predictions[AnalyzedPredictions.SENTIMENT_PREDICTION_CORRECT] = \
            matched_predictions.apply(AnalyzedPredictions.is_sentiment_correct, axis=1)
        matched_predictions[AnalyzedPredictions.FULL_PIPELINE_CORRECT] = \
            matched_predictions.apply(
                lambda match: match[AnalyzedPredictions.TARGET_EXTRACTION_CORRECT] and
                              match[AnalyzedPredictions.SENTIMENT_PREDICTION_CORRECT],
                axis=1)

    @staticmethod
    def match_predictions_to_labels(
            cluster_labels: List[LabeledCluster],
            predictions: SentimentTargets,
            matchers: List[Tuple[str, Callable[[LabeledCluster, LabeledSpan], bool]]]):
        predictions_as_labeled_spans = predictions.as_labeled_targets()
        scores = predictions.get_column_if_exists(column_name=TARGET_SCORE, default_value=1)

        matched_predictions = []
        for prediction, score in zip(predictions_as_labeled_spans, scores):
            matched_labels_list = []
            match_types = []
            for cluster_label in cluster_labels:
                for matcher in matchers:
                    if matcher[1](cluster_label, prediction):
                        matched_labels_list.append(cluster_label)
                        match_type = matcher[0]
                        match_types.append(match_type)
                        break
            matched_predictions.append({
                PREDICTION: prediction,
                LABELS: matched_labels_list,
                MATCH_TYPES: match_types,
                TARGET_SCORE: score
            })

        result = pandas.DataFrame(matched_predictions)
        if PREDICTION in result.columns:
            result['prediction.sentence_text'] = result[PREDICTION].apply(lambda x: x.text)
            result['prediction.target_text'] = result[PREDICTION].apply(lambda x: x.get_labeled_text())
            result['prediction.begin'] = result[PREDICTION].apply(lambda x: x.begin)
            result['prediction.end'] = result[PREDICTION].apply(lambda x: x.end)
            result['prediction.sentiment'] = result[PREDICTION].apply(lambda x: x.label)
        result = AnalyzedPredictions.expand_labels(result)
        return result

    @staticmethod
    def match_labels_to_predictions(
            cluster_labels: List[LabeledCluster],
            predictions: SentimentTargets,
            matchers: List[Tuple[str, Callable[[LabeledCluster, LabeledSpan], bool]]]):
        predictions = predictions.as_labeled_targets()

        matched_labels = []
        for cluster_label in cluster_labels:
            matched_predictions_list = []
            match_types = []
            for prediction in predictions:
                for matcher in matchers:
                    if matcher[1](cluster_label, prediction):
                        matched_predictions_list.append(prediction)
                        match_type = matcher[0]
                        match_types.append(match_type)
                        break
            matched_labels.append({
                LABELS: [cluster_label],
                PREDICTIONS: matched_predictions_list,
                MATCH_TYPES: match_types
            })
        result = pandas.DataFrame(matched_labels)

        result = AnalyzedPredictions.expand_predictions(result)
        result = AnalyzedPredictions.expand_labels(result)
        return result

    @staticmethod
    def expand_predictions(matched_frame):
        if PREDICTIONS not in matched_frame.columns:
            return
        matched_frame['# predictions'] = matched_frame[PREDICTIONS].apply(len)
        predictions_column = matched_frame[PREDICTIONS]
        result = []
        for row_i, predictions in enumerate(predictions_column):
            # work with dictionaries since its faster
            new_row = matched_frame.loc[row_i].to_dict()
            for prediction_i, prediction in enumerate(predictions):
                prefix = f'prediction_{prediction_i}'
                new_row[f'{prefix}.sentence_text'] = prediction.text
                new_row[f'{prefix}.target_text'] = prediction.get_labeled_text()
                new_row[f'{prefix}.begin'] = prediction.begin
                new_row[f'{prefix}.end'] = prediction.end
                new_row[f'{prefix}.sentiment'] = prediction.label
            result += [new_row]
        result = pandas.DataFrame(result)
        return result

    @staticmethod
    def expand_labels(matched_frame, labels_column_name=LABELS):
        if labels_column_name not in matched_frame:
            return

        labels_column = matched_frame[labels_column_name]
        matched_frame['# labels'] = labels_column.apply(len)
        matched_frame[MAJORITY_LABEL] = labels_column.apply(AnalyzedPredictions.get_majority_label)
        result = []
        for row_i, valid_target_groups in enumerate(labels_column):
            # work with dictionaries since its faster
            new_row = matched_frame.loc[row_i].to_dict()
            for group_i, target_group in enumerate(valid_target_groups):
                group_prefix = f'label_group_{group_i}'
                new_row[f'{group_prefix}.size'] = len(target_group.labeled_spans)
                new_row[f'{group_prefix}.sentence_text'] = target_group.text
                new_row[f'{group_prefix}.target_text'] = target_group.get_labeled_text()
                new_row[f'{group_prefix}.begin'] = target_group.span.left
                new_row[f'{group_prefix}.end'] = target_group.span.right
                new_row[f'{group_prefix}.sentiment'] = target_group.majority_label()
                new_row[f'{group_prefix}.label'] = target_group.get_aggregated_label()
                new_row[f'{group_prefix}.majority_count'] = target_group.get_aggregated_label().most_common_count
                new_row[f'{group_prefix}.is_unanimous'] = target_group.is_consistent_label()
                for label_i, target in enumerate(target_group.labeled_spans):
                    label_prefix = f'{group_prefix}.label_{label_i}'
                    new_row[f'{label_prefix}.target_text'] = target.get_labeled_text()
                    new_row[f'{label_prefix}.begin'] = target.begin
                    new_row[f'{label_prefix}.end'] = target.end
                    new_row[f'{label_prefix}.sentiment'] = target.label
            result += [new_row]
        result = pandas.DataFrame(result)
        return result

    def get_stats(self):
        result = pandas.Series(self.stats, name=self.name)
        return result

    def get_stat(self, *, stat_name=None, task_name=None, label=None, metric=None):
        if stat_name and task_name:
            raise RuntimeError('Both stat_name and task_name are set')
        if not stat_name and not task_name:
            raise RuntimeError('Missing parameter stat_name or task_name')
        if task_name:
            stat_name = get_measure_name(task_name, label=label, metric=metric)
        return self.stats[stat_name]

    def as_tsa_labels(self):
        labeled_frame = self.labeled[SentimentTargets.KEY_COLUMNS + TsaLabels.LABEL_COLUMNS].copy()
        labeled_frame.drop_duplicates(inplace=True)
        return TsaLabels(labeled_frame=labeled_frame)

    def to_csv(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.predictions_with_labels_frame.to_csv(self.get_report_path(directory, report_name='all'))
        self.labeled.to_csv(self.get_report_path(directory, report_name='labeled'))

        self.matched_predictions.to_csv(self.get_report_path(directory, report_name='matched_predictions'))
        unlabeled_predictions = self.matched_predictions[self.matched_predictions['is_unlabeled']].copy()
        unlabeled_predictions.dropna(how='all', axis=1, inplace=True)
        unlabeled_predictions.to_csv(self.get_report_path(directory, report_name='unlabeled_predictions'))

        self.matched_labels.to_csv(self.get_report_path(directory, report_name='matched_labels'))
        is_covered = self.matched_labels['is_covered_label']
        unmatched_labels = self.matched_labels[~is_covered].copy()
        unmatched_labels.dropna(how='all', axis=1, inplace=True)
        unmatched_labels.to_csv(self.get_report_path(directory, report_name='unmatched_labels'))

    @staticmethod
    def get_report_path(report_directory, report_name):
        Path(report_directory).mkdir(exist_ok=True, parents=True)
        return os.path.join(report_directory, f'{report_name}.csv')

    def stats_at_threshold(self, task_name=TARGETED_SENTIMENT_ANALYSIS):
        is_correct_measure_name = get_measure_name(task_name=task_name, metric=IS_CORRECT)
        sorted_predictions = self.matched_predictions.sort_values(by=TARGET_SCORE, ascending=False)
        sorted_predictions = sorted_predictions[~sorted_predictions[IS_IGNORE_LABEL]]
        scores = sorted_predictions[TARGET_SCORE]
        is_correct = sorted_predictions[is_correct_measure_name].reset_index(drop=True)
        commulative_num_correct = is_correct.cumsum()
        precision = commulative_num_correct / range(1, len(is_correct)+1)
        recall = commulative_num_correct / self.stats[NUM_LABELED_CLUSTERS]
        result = sorted_predictions.reset_index(drop=True)[[PREDICTION, LABELS, MATCH_TYPES, is_correct_measure_name, IS_IGNORE_LABEL]]
        result['num correct'] = commulative_num_correct
        result[TARGET_SCORE] = scores.reset_index(drop=True)
        result[PRECISION] = precision
        result[RECALL] = recall
        result[F1] = result.apply(lambda x:  statistics.harmonic_mean([x[PRECISION], x[RECALL]]), axis=1)
        result[F05] = result.apply(lambda x: compute_f05(x[PRECISION], x[RECALL]), axis=1)
        return result


