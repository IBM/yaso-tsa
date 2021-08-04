# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0


import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

import nltk
import pandas as pd
import xml.etree.ElementTree as ET


def txt_sha1(txt):
    sha1 = hashlib.sha1()
    sha1.update(txt.encode())
    return sha1.hexdigest()


def resolve_path(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ValueError(f'Path does not exist: {os.path.abspath(path)}.'
                         f' Please refer to README file for instructions.')
    return path


def param_amazon(args):
    param = args.amazon
    return None if param is None else {'reviews_file': resolve_path(param)}


def param_sst(args):
    param = args.sst
    return None if param is None else {'sst_dir': resolve_path(param)}


def param_opinosis(args):
    param = args.opinosis
    return None if param is None else {'topics_dir': resolve_path(param)}


def param_semeval_14(args):
    param = args.semeval
    if param is None:
        return param

    test_dir = resolve_path(param)
    laptops = None if test_dir is None else os.path.join(test_dir, 'Laptops_Test_Gold.xml')
    restaurants = None if test_dir is None else os.path.join(test_dir, 'Restaurants_Test_Gold.xml')
    if test_dir is not None:
        if not os.path.exists(laptops):
            raise ValueError(f'SemEval14 laptops file does not exist: {os.path.abspath(laptops)}')
        if not os.path.exists(restaurants):
            raise ValueError(f'SemEval14 restaurants file does not exist: {os.path.abspath(restaurants)}')
    return {'xml_files': [laptops, restaurants]}


def restore_amazon(hashes, reviews_file):
    hash_to_restored_sentence = {}
    with open(reviews_file, encoding='utf-8') as f:
        for line in f:
            file_record = json.loads(line)
            sentences = nltk.sent_tokenize(file_record['review_body'])
            for sentence in sentences:
                sentence = sentence.strip()
                hash_str = txt_sha1(sentence)
                if hash_str in hashes:
                    restored_sentence = file_record.copy()
                    restored_sentence['text'] = sentence
                    hash_to_restored_sentence[hash_str] = restored_sentence
                if len(hash_to_restored_sentence) == len(hashes):
                    return hash_to_restored_sentence

    return hash_to_restored_sentence


def restore_sst(hashes, sst_dir):
    def read_as_dict(file, delim, key_col, val_col, **read_kwargs):
        df = pd.read_csv(file, delimiter=delim, **read_kwargs)
        return {key: val for key, val in zip(df[key_col], df[val_col])}

    sent_split = read_as_dict(os.path.join(sst_dir, 'datasetSplit.txt'), ',', 'sentence_index', 'splitset_label')
    sentences = read_as_dict(os.path.join(sst_dir, 'datasetSentences.txt'), '\t', 'sentence_index', 'sentence')

    hash_to_restored_sentence = {}
    for sent_id, sentence in sentences.items():
        if sent_split[sent_id] == 2:
            sentence = sentence.strip().replace('-LRB- ', '(').replace(' -RRB-', ')')
            hash_str = txt_sha1(sentence)
            if hash_str in hashes:
                hash_to_restored_sentence[hash_str] = {
                    'text': sentence
                }
            if len(hash_to_restored_sentence) == len(hashes):
                return hash_to_restored_sentence
    return hash_to_restored_sentence


def restore_opinosis(hashes, topics_dir):
    hash_to_restored_sentence = {}
    topic_files = glob.glob(os.path.join(topics_dir, '*.txt.data'))
    for topic_file in topic_files:
        with open(topic_file, errors='ignore') as f:
            for line in f:
                sentence = line.strip()
                hash_str = txt_sha1(sentence)
                if hash_str in hashes:
                    topic = Path(Path(topic_file).stem).stem
                    hash_to_restored_sentence[hash_str] = {
                        'text': sentence,
                        'topic': topic
                    }
                if len(hash_to_restored_sentence) == len(hashes):
                    return hash_to_restored_sentence
    return hash_to_restored_sentence


def restore_semeval_14(hashes, xml_files):
    hash_to_restored_sentence = {}
    for xml_file in xml_files:
        root = ET.parse(xml_file).getroot()
        for sentence_tag in root.findall('sentence'):
            sentence = sentence_tag.find('text').text.strip()
            hash_str = txt_sha1(sentence)
            if hash_str in hashes:
                hash_to_restored_sentence[hash_str] = {
                    'text': sentence
                }
            if len(hash_to_restored_sentence) == len(hashes):
                return hash_to_restored_sentence
    return hash_to_restored_sentence


def restore_text(data, out_json, src_param):
    def _get_hash(r):
        if 'review_id' in r:
            return r['review_id'], r['text_hash']
        else:
            return r['text_hash']

    def _remove_hash(r):
        r.pop('text_hash')
        if 'review_id' in r:
            r.pop('review_id')

    src_hashes = {}
    for output_record in data:
        if output_record['text'] is None:
            src = output_record['source']
            src_hashes[src] = src_hashes.get(src, []) + [_get_hash(output_record)]

    src_missing_hashes = {}
    for src, hashes in src_hashes.items():
        kwargs = src_param[src]
        if kwargs is None:
            src_hashes[src] = None
            print(f'No input argument provided for source {src}. Its {len(hashes)} sentences will not be restored.')
        else:
            print(f'Restoring {len(hashes)} sentences from {src}')
            hash2txt = RESTORE_FUNCTIONS[src]['restore_fun'](hashes, **kwargs)
            src_hashes[src] = hash2txt
            src_missing_hashes[src] = [h for h in hashes if h not in hash2txt]

    out_data = []
    for output_record in data:
        restored = False
        if output_record['text'] is None:
            hash_to_restored_sentences = src_hashes[output_record['source']]
            if hash_to_restored_sentences is not None:
                # restored texts exists for this source
                text_hash = _get_hash(output_record)
                if text_hash in hash_to_restored_sentences:
                    # restored text found for this text
                    restored_sentence = hash_to_restored_sentences[text_hash]
                    output_record.update(restored_sentence)
                    _remove_hash(output_record)
                    out_data.append(output_record)
                    restored = True

        if not restored:
            out_data.append(output_record)

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f'{len(out_data)} sentences with restored text saved to {os.path.abspath(out_json)}')
    for src, missing in src_missing_hashes.items():
        if len(missing) > 0:
            print(f'{len(missing)} sentence hash codes not resolved for source {src}:\n\t{missing}')


def prepare_src_param(in_json, args):
    with open(in_json, encoding='utf-8') as f:
        data = json.load(f)
    src_param = {rec['source']: None for rec in data}
    for src in src_param:
        if src in RESTORE_FUNCTIONS:
            src_param[src] = RESTORE_FUNCTIONS[src]['param_fun'](args)

    return data, src_param


RESTORE_FUNCTIONS = {
    'Amazon': {
        'param_fun': param_amazon,
        'restore_fun': restore_amazon,
    },
    'SST2': {
        'param_fun': param_sst,
        'restore_fun': restore_sst,
    },
    'Opinosis': {
        'param_fun': param_opinosis,
        'restore_fun': restore_opinosis,
    },
    'SemEval14': {
        'param_fun': param_semeval_14,
        'restore_fun': restore_semeval_14,
    },
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restore sentences of the YASO evaluation set.')
    parser.add_argument('--amazon', help='path to file dataset_en_test.json')
    parser.add_argument('--sst', help='path to directory stanfordSentimentTreebank')
    parser.add_argument('--opinosis', help='path to directory topics')
    parser.add_argument('--semeval', help='path to directory ABSA_Gold_TestData')

    args = parser.parse_args()

    in_file = 'yaso_hidden.json'
    out_file = 'yaso.json'

    in_data, source_param = prepare_src_param(in_file, args)
    restore_text(in_data, out_file, source_param)
