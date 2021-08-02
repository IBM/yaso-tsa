# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import logging

from yaso_tsa.Analysis.AnalzyedPredictions import AnalyzedPredictions, TARGETED_SENTIMENT_ANALYSIS, PRECISION, RECALL, F1
from yaso_tsa.infra.TsaData import TsaData
from yaso_tsa.infra.TsaLabels import TsaLabels

logging.basicConfig(format='[%(threadName)s] %(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate TSA predictions.')
    parser.add_argument('--predictions_path', help='path to predictions json file')
    parser.add_argument('--labels_path', help='path to labels json file')

    args = parser.parse_args()

    predictions = TsaData.read_json(path=args.predictions_path)
    tsa_labels = TsaLabels.read_json(path=args.labels_path)
    analysis = AnalyzedPredictions(
        all_predictions=predictions.get_sentiment_targets(),
        labeled_data=tsa_labels
    )

    def report_metric(metric):
        logging.info(f'{metric}={analysis.get_stat(TARGETED_SENTIMENT_ANALYSIS, metric=metric)}')

    report_metric(PRECISION)
    report_metric(RECALL)
    report_metric(F1)
