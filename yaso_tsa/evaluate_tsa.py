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

LABELS_PATH = '--labels_path'
PREDICTIONS_PATH = '--predictions_path'


def main():
    parser = argparse.ArgumentParser(description='Evaluate TSA predictions.')
    parser.add_argument(PREDICTIONS_PATH, help='path to predictions json file', required=True)
    parser.add_argument(LABELS_PATH, help='path to labels json file', required=True)
    parser.add_argument('--extend_labels',
                        help='extend the tsa labels via rules (default: false)',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    predictions = TsaData.read_json(path=args.predictions_path)
    tsa_labels = TsaLabels.read_json(path=args.labels_path)
    logging.info(f'Loaded labeled data: {tsa_labels}')
    if args.extend_labels:
        tsa_labels = tsa_labels.extend_labels()
        logging.info(f'Extended labeled data: {tsa_labels}')
    analysis = AnalyzedPredictions(
        tsa_data=predictions,
        labeled_data=tsa_labels
    )

    def report_metric(metric):
        logging.info(f'{metric}='
                     f'{analysis.get_stat(task_name=TARGETED_SENTIMENT_ANALYSIS, metric=metric)}')

    report_metric(PRECISION)
    report_metric(RECALL)
    report_metric(F1)


if __name__ == '__main__':
    main()
