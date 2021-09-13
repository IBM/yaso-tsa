# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from unittest import TestCase

from yaso_tsa.infra.LabeledSpan import LabeledSpan

import pandas


class TestLabeledSpan(TestCase):

    def test_span_as_interval(self):
        labeled_span = LabeledSpan(text='This is a sentence', begin=0, end=3, label='label_value')
        self.assertEqual(labeled_span.span_as_interval(), pandas.Interval(0, 3, closed='both'))

    def test_overlaps(self):
        labeled_span = LabeledSpan(text='This is a sentence', begin=0, end=3, label='label_value')
        overlapping = LabeledSpan(text='This is a sentence', begin=3, end=6, label='label_value')
        self.assertTrue(labeled_span.overlaps(overlapping))
        self.assertTrue(overlapping.overlaps(labeled_span))

        non_overlapping = LabeledSpan(text='This is a sentence', begin=4, end=7, label='label_value')
        self.assertFalse(labeled_span.overlaps(non_overlapping))
        self.assertFalse(non_overlapping.overlaps(labeled_span))

        same_span_other_sentence = LabeledSpan(text='Different text', begin=0, end=3, label='label_value')
        self.assertFalse(labeled_span.overlaps(same_span_other_sentence))
        self.assertFalse(same_span_other_sentence.overlaps(labeled_span))

    def test_is_same_span(self):
        labeled_span = LabeledSpan(text='This is a sentence', begin=0, end=3, label='label_value')
        same_span = LabeledSpan(text='This is a sentence', begin=0, end=3, label='label_value')

        self.assertTrue(labeled_span.is_same_span(same_span))
        self.assertTrue(same_span.is_same_span(labeled_span))

        other_span = LabeledSpan(text='This is a sentence', begin=1, end=3, label='label_value')
        self.assertFalse(labeled_span.is_same_span(other_span))
        self.assertFalse(other_span.is_same_span(labeled_span))

        same_span_other_sentence = LabeledSpan(text='Different text', begin=0, end=3, label='label_value')
        self.assertFalse(labeled_span.is_same_span(same_span_other_sentence))
        self.assertFalse(same_span_other_sentence.is_same_span(labeled_span))

    def test_get_labeled_text(self):
        text = 'This is a sentence'
        begin = 0
        end = 4
        labeled_span = LabeledSpan(text=text, begin=begin, end=end, label='label_value')
        self.assertEqual(labeled_span.get_labeled_text(), text[begin:end])
        self.assertEqual(labeled_span.get_labeled_text(), 'This')
