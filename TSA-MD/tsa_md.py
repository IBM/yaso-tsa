from yaso_tsa.infra.TsaLabels import TsaLabels

if __name__ == '__main__':
    '''
    This script demonstrates how to read the TSA-MD data files into TsaLabels objects.
    '''
    dev = TsaLabels.read_json(path='TSA-MD.dev.json')
    train = TsaLabels.read_json(path='TSA-MD.train.json')
    print(f"TSA-MD training data contains: {train}")
    print(f"TSA-MD training development contains: {dev}")

    # The above should print:
    # TSA-MD training data contains: <TsaLabels labeled: 1212, sentences: 761>
    # TSA-MD training development contains: <TsaLabels labeled: 311, sentences: 191>
