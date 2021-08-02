# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import pandas

from yaso_tsa.infra.CategoricalLabel import CategoricalLabel


@dataclass
class LabeledSpan:
    text: str
    begin: int
    end: int
    label: CategoricalLabel

    def __repr__(self):
        return f"<LabeledSpan (" \
                    f"labeled_text={self.get_labeled_text()}, " \
                    f"label={self.label}, span={self.begin}:{self.end}, text={self.text})>"

    def span_as_interval(self):
        return pandas.Interval(self.begin, self.end, closed='both')

    def overlaps(self, other):
        return self.text == other.text and self.span_as_interval().overlaps(other.span_as_interval())

    def is_same_span(self, other):
        return self.begin == other.begin and self.end == other.end and self.text == other.text

    def get_labeled_text(self):
        return self.text[self.begin:self.end]

