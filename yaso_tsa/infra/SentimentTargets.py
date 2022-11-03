# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import json
import pandas as pd
import logging

TARGETS = 'target_mentions'
SOURCE = 'source'
SENTENCE_TEXT = 'text'
TARGET_TEXT = 'target_text'
TARGET_BEGIN = 'location_begin'
TARGET_END = 'location_end'
TARGET_SENTIMENT = 'sentiment'
TARGET_SCORE = 'confidence'


class SentimentTargets:

    KEY_COLUMNS = [TARGET_TEXT, SENTENCE_TEXT, TARGET_BEGIN, TARGET_END]
    MANDATORY_COLUMNS = KEY_COLUMNS + [TARGET_SENTIMENT]

    @staticmethod
    def read_json(path, meta_fields=[]):
        logging.debug(f'Reading from "{path}"')
        with open(path, encoding='utf8') as json_file:
            try:
                json_contents = json.load(json_file)
                sentiment_targets = pd.json_normalize(
                    json_contents,
                    record_path=[TARGETS],
                    # add a record_prefix otherwise the target text and sentence text fields collide (both named 'text)
                    record_prefix='target_',
                    meta_prefix='',
                    meta=['text'] + meta_fields)
            except Exception as e:
                raise RuntimeError(f'Cannot read from "{path}"', e)
            sentiment_targets.rename(
                columns=lambda x: x.replace('target_', '') if x != TARGET_TEXT else x,
                inplace=True
            )
            sentiment_targets.rename(
                columns={
                    'location.begin': TARGET_BEGIN,
                    'location.end': TARGET_END
                },
                inplace=True
            )
            if not sentiment_targets.empty:
                sentiment_targets[TARGET_BEGIN] = sentiment_targets[TARGET_BEGIN].astype(int)
                sentiment_targets[TARGET_END] = sentiment_targets[TARGET_END].astype(int)
            result = SentimentTargets(frame=sentiment_targets)
            result = result.update_sentiment(old='neutral', new='none')
            return result

    def __init__(self, frame=None):
        if frame is not None:
            if frame.empty:
                self.frame = pd.DataFrame(columns=self.MANDATORY_COLUMNS)
            else:
                self.frame = frame.copy()
                missing_columns = [column_name for column_name in self.MANDATORY_COLUMNS if column_name not in self.frame.columns]
                if missing_columns:
                    raise ValueError(f'Missing "{missing_columns}" columns from frame.')
        else:
            self.frame = pd.DataFrame(columns=self.MANDATORY_COLUMNS)

    def __repr__(self):
        return f"<SentimentTargets , " \
               f"targets: {self.get_num_targets()}, sentences: {len(self.get_sentences())}>"

    def copy(self):
        return SentimentTargets(sentiment_targets=self.frame.copy())

    def get_frame(self):
        return self.frame

    def get_column_if_exists(self, column_name, default_value):
        return self.get_frame().get(
            key=column_name,
            default=pd.Series(
                default_value,
                index=range(self.get_num_targets())
            )
        )

    def as_labeled_targets(self):
        from ..infra.LabeledTarget import LabeledTarget
        return self.frame.apply(
            lambda row: LabeledTarget.create(row=row, index_label=TARGET_SENTIMENT), axis=1)

    def as_labels(self):
        from ..infra.TsaLabels import TsaLabels
        return TsaLabels(frame=self.get_frame())

    def write_sentences(self, path):
        sentences = self.get_sentences()
        pd.Series(sentences).to_csv(path, header=True)

    def to_csv(self, path,
              extra_columns=None,
              with_sentiment=False,
              shuffle=True):

        output_columns = SentimentTargets.KEY_COLUMNS.copy()
        if with_sentiment:
            output_columns += [TARGET_SENTIMENT]
        if extra_columns is not None:
            output_columns = output_columns + extra_columns

        if shuffle:
            output = self.shuffle()
        else:
            output = self

        output = output.get_frame().copy()
        output.to_csv(path, columns=output_columns, index=False)

    def shuffle(self):
        result = self.frame.sample(frac=1, random_state=SentimentTargets.get_random_state())
        return SentimentTargets(frame=result)

    def add(self, other):
        result = self.frame.append(other.frame)
        return SentimentTargets(frame=result)

    def add_property(self, property_name, property_value):
        self.frame[property_name] = property_value

    def sample_sentences(self, num_sentences):
        sentences = self.get_sentences()
        sampled_sentences = sentences.sample(n=num_sentences, random_state=SentimentTargets.get_random_state())
        sampled = self.select_sentences(sentences=sampled_sentences)
        return sampled

    def select_sentences(self, sentences):
        result = self.frame[self.frame[SENTENCE_TEXT].isin(sentences)]
        return SentimentTargets(frame=result)

    def select_targets(self, required_sentiment=None, condition=None):
        if required_sentiment is not None:
            return SentimentTargets(
                frame=self.frame[self.frame[TARGET_SENTIMENT].isin(required_sentiment)])
        elif condition:
            return SentimentTargets(
                frame=self.frame[self.frame.apply(condition, axis=1)]
            )
        else:
            return self

    def update_sentiment(self, old, new):
        is_old_sentiment = self.frame[TARGET_SENTIMENT] == old
        self.frame.loc[is_old_sentiment, TARGET_SENTIMENT] = new
        return self

    def remove_sentences(self, sentences):
        result = self.frame[~self.frame[SENTENCE_TEXT].isin(sentences)]
        return SentimentTargets(sentiment_targets=result)

    def sample_targets(self, num_targets_to_sample, required_sentiment_in_samples=None):
        with_required_sentiment = self.select_targets(required_sentiment=required_sentiment_in_samples).frame
        if required_sentiment_in_samples is None:
            required_sentiment_in_samples = 'All'
        num_available_targets = len(with_required_sentiment)
        if num_available_targets == 0:
            result = SentimentTargets(sentiment_targets=pd.DataFrame(columns=SentimentTargets.columns))
        else:
            num_targets_to_sample = min(num_targets_to_sample, num_available_targets)
            sampled = with_required_sentiment.sample(n=num_targets_to_sample)
            result = SentimentTargets(sentiment_targets=sampled)
        logging.info(f'Sampled {result.get_num_targets()} targets with sentiment "{required_sentiment_in_samples}" '
                     f'from {num_available_targets} available targets.')
        return result

    def group_and_sample_targets(self, group_column, num_targets_to_sample):
        result = self.frame.groupby(group_column).apply(
            lambda x:
                SentimentTargets(sentiment_targets=x).
                sample_targets(num_targets_to_sample=num_targets_to_sample)
                .frame
        )
        return SentimentTargets(sentiment_targets=result)

    def remove_targets(self, targets_to_remove):
        from labeling.collected_utils import remove_rows
        num_targets_before_removal = self.get_num_targets()
        targets_after_removal = remove_rows(
            source=self.frame,
            items_to_remove=targets_to_remove.frame,
            key_columns=SentimentTargets.KEY_COLUMNS)
        targets_after_removal = SentimentTargets(sentiment_targets=targets_after_removal)
        num_targets_after_removal = targets_after_removal.get_num_targets()
        logging.info(f'Num targets: after removal: {num_targets_after_removal}, '
                     f'before removal: {num_targets_before_removal}')
        return targets_after_removal

    def get_sentences(self):
        sentences = pd.unique(self.frame[SENTENCE_TEXT])
        return list(pd.Series(sentences))

    def get_num_sentences(self):
        return len(self.get_sentences())

    def unique(self, use_sentiment=False):
        key_columns = SentimentTargets.KEY_COLUMNS.copy()
        if use_sentiment:
            key_columns += [TARGET_SENTIMENT]
        return SentimentTargets(frame=self.frame.drop_duplicates(subset=key_columns))

    def get_num_targets(self):
        return len(self.frame)

    def log_num_sentence_with_predictions(self, description=None):
        num_sentences = len(self.get_sentences())
        if description is not None:
            description = f'{description}: '
        else:
            description = ""
        logging.info(f'{description}Contains {num_sentences} unique sentences with {self.get_num_targets()} targets.')

    def log_sentiment_histogram(self, description=""):
        if description:
            description = f'{description}: '
        logging.info(f'{description}Sentiment histogram\n'
                     f'{self.frame[TARGET_SENTIMENT].value_counts(normalize=True)}')

    def get_sentiment_counts(self):
        return self.frame[TARGET_SENTIMENT].value_counts()

    def remove_labeled(self, labeled_sentiment_targets):
        is_labeled = self.frame.apply(
            lambda x: labeled_sentiment_targets.is_labeled(x['target_text'], x['text']), 1)
        num_labeled = sum(is_labeled)
        logging.info(f'num_labeled: {num_labeled} targets out of {self.get_num_targets()}')
        result_frame = self.frame[~is_labeled]
        return SentimentTargets(sentiment_targets=result_frame)

    @staticmethod
    def get_random_state():
        return 888
