# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import json
import logging
import pandas as pd

from yaso_tsa.infra.SentimentTargets import SentimentTargets, SENTENCE_TEXT, TARGET_TEXT, TARGET_BEGIN, TARGET_END, \
    TARGET_SENTIMENT


class TsaData:

    '''
    A collection of sentences and targets within those sentences.
    Some of the sentences may not have marked targets in them.
    Each target has a span and a label.
    '''

    @staticmethod
    def read_xml(path):

        """
        Read XML of this format:
        	<sentence>
		        <text>The decor is not special at all but their food and amazing prices make up for it.</text>
		        <aspectTerms>
			        <aspectTerm from="4" polarity="negative" term="decor" to="9"/>
			        <aspectTerm from="42" polarity="positive" term="food" to="46"/>
			    <aspectTerm from="59" polarity="positive" term="prices" to="65"/>
		    </aspectTerms>
	        </sentence>
        :param path:
        :return:
        """

        from xml.dom import minidom
        mydoc = minidom.parse(path)
        sentences = mydoc.getElementsByTagName('sentence')
        sentiment_targets = pd.DataFrame()
        sentence_texts = []
        for sentence in sentences:
            sentence_text = sentence.getElementsByTagName('text')
            if len(sentence_text) != 1:
                raise RuntimeError(f'Unexpected {len(sentence_text)} sentence text elements')
            sentence_text = sentence_text[0].firstChild.data
            sentence_texts += [sentence_text]
            targets = sentence.getElementsByTagName('aspectTerm')
            for target in targets:
                sentiment_targets = sentiment_targets.append({
                    SENTENCE_TEXT: sentence_text,
                    TARGET_BEGIN: target.attributes['from'].value,
                    TARGET_END: target.attributes['to'].value,
                    TARGET_SENTIMENT: target.attributes['polarity'].value,
                    TARGET_TEXT: target.attributes['term'].value
                }, ignore_index=True)
        result = TsaData(
            sentences=pd.DataFrame({SENTENCE_TEXT: sentence_texts}),
            sentiment_targets=SentimentTargets(frame=sentiment_targets)
        )
        logging.info(f'Loaded {result} from "{path}"')
        return result

    @staticmethod
    def read_json(path, meta_fields=[]):
        sentiment_targets = SentimentTargets.read_json(path=path, meta_fields=meta_fields)
        with open(path, "rt", encoding="utf-8") as json_file:
            json_contents = json.load(json_file)
            sentences = pd.DataFrame()
            for meta_field in [SENTENCE_TEXT] + meta_fields:
                sources = [labeled_sentence[meta_field] for labeled_sentence in json_contents]
                sentences[meta_field] = sources
            logging.debug(f'Found {len(sentences)} sentences in json "{path}"')
            return TsaData(sentiment_targets=sentiment_targets, sentences=sentences, name=path)

    @staticmethod
    def read_jsons(files, verbose=False, meta_fields=[]):
        result = TsaData()
        for file in files:
            data = TsaData.read_json(path=file, meta_fields=meta_fields)
            if verbose:
                logging.info(f'Loaded from "{file}": {data}')
            result = result.add(data)

        if verbose:
            logging.info(f'Contents loaded from all files: {result}')
        return result

    def __init__(
        self,
        sentiment_targets: SentimentTargets = None,
        sentences=pd.DataFrame(),
        name=None
    ):
        self.__sentiment_targets = sentiment_targets if sentiment_targets is not None else SentimentTargets()
        if sentences.empty:
            self.__sentences = pd.DataFrame(columns=[SENTENCE_TEXT])
        else:
            if SENTENCE_TEXT not in sentences.columns:
                raise ValueError('Missing column from sentences.')
            self.__sentences = sentences
        self.__name = name

    def __repr__(self):
        return f"<TsaData {self.get_name()}, " \
               f"targets: {self.__sentiment_targets}, # sentences: {len(self.get_sentences())}>"

    def get_name(self):
        return self.__name if self.__name else 'unnamed'

    def get_sentiment_targets(self):
        return self.__sentiment_targets

    def get_sentences_frame(self):
        return self.__sentences

    def select_targets(self, required_sentiment=None):
        self.__sentiment_targets = self.__sentiment_targets.select_targets(required_sentiment)

    def drop_duplicates(self):
        self.__sentiment_targets = self.__sentiment_targets.unique(use_sentiment=True)

    def get_sentences(self):
        return list(self.__sentences[SENTENCE_TEXT])

    def select_first_sentences(
            self, num_to_select=None, percentage=None, select_from_start=True):
        sentences = self.get_sentences()
        num_sentences = len(sentences)
        if not num_to_select:
            num_to_select = int(num_sentences * percentage)
        num_to_select = min(num_to_select, num_sentences)
        if select_from_start:
            selected = sentences[:num_to_select]
        else:
            selected = sentences[-num_to_select:]  # select from end
        return self.select_sentences(sentences=selected)

    def select_sentences(self, sentences, new_name=""):
        selected_targets = self.__sentiment_targets.select_sentences(sentences)
        selected_sentences = self.__sentences[self.__sentences[SENTENCE_TEXT].isin(sentences)]
        return TsaData(sentiment_targets=selected_targets, sentences=selected_sentences, name=new_name)

    def shuffle(self):
        shuffled_targets = self.__sentiment_targets.shuffle()
        shuffled_sentences = self.__sentences.sample(frac=1, random_state=SentimentTargets.get_random_state())
        return TsaData(sentiment_targets=shuffled_targets, sentences=shuffled_sentences)

    def to_json(self, path, to_dict=None, shuffle=False):

        # convert one sentence to a dictionary representation
        def single_sentence_as_dictionary(single_text_sentiment_targets):
            return {
                'text': single_text_sentiment_targets.name,
                'targets': single_text_sentiment_targets.apply(one_target_as_dictionary, axis=1)
            }

        def one_target_as_dictionary(single_target):
            result = {
                'text': single_target[TARGET_TEXT],
                'location': {
                    'begin': int(single_target[TARGET_BEGIN]),
                    'end': int(single_target[TARGET_END])
                },
                'sentiment': single_target[TARGET_SENTIMENT],
            }
            if to_dict:
                result.update(to_dict(single_target))
            return result

        sentences = self.get_sentiment_targets().get_frame()\
            .groupby(SENTENCE_TEXT).apply(single_sentence_as_dictionary)
        sentences_without_targets = self.get_sentences_without_targets()
        for sentence in sentences_without_targets:
            sentences[sentence] = {
                'text': sentence,
                'targets': []
            }
        if shuffle:
            logging.info("Shuffling output")
            sentences = sentences.sample(frac=1)
        sentences.to_json(path, orient='records', double_precision=2, indent=2)
        num_sentences_without_targets = len(sentences_without_targets)
        logging.info(f'{len(sentences)} sentences (without targets: {num_sentences_without_targets}) '
                     f'written to "{path}"')

    def get_sentences_with_predictions(self):
        return self.__sentiment_targets.get_sentences()

    def get_sentences_without_targets(self):
        sentences_with_predictions = self.get_sentences_with_predictions()
        all_sentences = self.get_sentences()
        result = set(all_sentences).difference(sentences_with_predictions)
        return result

    def add(self, other):
        sentiment_targets = self.get_sentiment_targets().add(other.get_sentiment_targets())
        sentiment_targets = sentiment_targets.unique()
        sentences = self.__sentences.append(other.__sentences)
        sentences = sentences.drop_duplicates(ignore_index=True)
        return TsaData(sentiment_targets=sentiment_targets, sentences=sentences)

    def copy(self):
        return TsaData(
            sentiment_targets=self.get_sentiment_targets().copy(),
            sentences=self.__sentences.copy(),
            name=self.get_name()
        )

    def unique_targets(self):
        return TsaData(
            self.get_sentiment_targets().unique(use_sentiment=True),
            sentences=self.__sentences.copy(),
            name=self.get_name()
        )

    @staticmethod
    def sub(file, files_to_remove):
        if not isinstance(files_to_remove, list):
            files_to_remove = [files_to_remove]
        excluded_sentences = TsaData.read_jsons(files=files_to_remove).get_sentences()
        result = TsaData(json_path=file)
        selected_sentences = [sentence for sentence in result.get_sentences() if sentence not in excluded_sentences]
        result = result.select_sentences(selected_sentences)
        return result
