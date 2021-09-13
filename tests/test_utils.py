# © Copyright IBM Corporation 2021.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os


def get_test_data_path():
    return os.path.join('data', 'test_data.json')


def get_test_data_path_2():
    return os.path.join('data', 'test_data_2.json')


def get_test_labels_path():
    return os.path.join('data', 'test_labels.json')


def get_test_labels_with_the_path():
    return os.path.join('data', 'test_labels_with_THE.json')


def get_test_xml_data_path():
    return os.path.join('data', 'test_data.xml')
