# YASO: A Targeted Sentiment Analysis Evaluation Dataset for Open-Domain Reviews

<!-- Not always needed, but a scope helps the user understand in a short sentance like below, why this repo exists -->
## Scope

This repository contains:

 (1) The YASO evaluation dataset for targeted sentiment analysis (TSA).

 (2) Code for evaluating the output of TSA systems on YASO.  

## Usage

**Data**: Some of the sentences annotated in YASO are taken from other datasets that cannot be re-distributed in clear text. To obtain the original texts please follow the instructions [here](yaso_tsa/data/README.md).

**Evaluation code**: 

<ins>Installation</ins>

Using pip:
```
pip install git+ssh://git@github.com/IBM/yaso-tsa.git#egg=yaso-tsa
```

Alternatively, you can first clone the code, and install the requirements: 

```commandline
1. git clone git@github.com:IBM/yaso-tsa.git
2. cd yaso-tsa/yaso_tsa
3. pip install -r requirements.txt
```

<ins>Running an evaluation</ins>

Use the module `yaso_tsa.evaluate_tsa`.
 
For example, run the following command from the main directory of the repository:  

```commandline
python -m yaso_tsa.evaluate_tsa --predictions_path tests/data/test_data.json --labels_path tests/data/test_labels.json
```

The expected output should be similar to:

```text
[MainThread] 2021-09-13:16:37:15,137 INFO     [evaluate_tsa.py:34] Loaded labeled data: <TsaLabels labeled: 4, sentences: 3>
[MainThread] 2021-09-13:16:37:15,190 INFO     [evaluate_tsa.py:44] precision=0.6666666666666666
[MainThread] 2021-09-13:16:37:15,190 INFO     [evaluate_tsa.py:44] recall=0.6666666666666666
[MainThread] 2021-09-13:16:37:15,190 INFO     [evaluate_tsa.py:44] F1=0.6666666666666666
```

## Citing YASO

If you are using YASO in a publication, please cite the following paper:

Matan Orbach, Orith Toledo-Ronen, Artem Spector, Ranit Aharonov, Yoav Katz and Noam Slonim. 2021.
[YASO: A Targeted Sentiment Analysis Evaluation Dataset for Open-Domain Reviews](https://arxiv.org/abs/2012.14541). EMNLP.  

## Contributing

This project welcomes external contributions, if you would like to contribute please see further instructions [here](CONTRIBUTING.md)

Pull requests are very welcome! Make sure your patches are well tested.
Ideally create a topic branch for every separate change you make. For
example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## Changelog

<!-- A Changelog allows you to track major changes and things that happen, https://github.com/github-changelog-generator/github-changelog-generator can help automate the process -->
Major changes are documented [here](CHANGELOG.md).

<!-- The following are OPTIONAL, but strongly suggested to have in your repository. 
* [dco.yml](.github/dco.yml) - This enables DCO bot for you, please take a look https://github.com/probot/dco for more details.
* [travis.yml](.travis.yml) - This is a example `.travis.yml`, please take a look https://docs.travis-ci.com/user/tutorial/ for more details.
-->

<!-- A notes section is useful for anything that isn't covered in the Usage or Scope. Like what we have below. -->
## Notes

<!--
**NOTE: This repository has been configured with the [DCO bot](https://github.com/probot/dco).
When you set up a new repository that uses the Apache license, you should
use the DCO to manage contributions. The DCO bot will help enforce that.
Please contact one of the IBM GH Org stewards.**
-->

If you have any questions or issues you can create a new [issue here][issues].

## License

This code is distributed under Apache License 2.0. If you would like to see the detailed LICENSE click [here](LICENSE).

## Authors

The YASO dataset was collected by Matan Orbach, Orith Toledo-Ronen, Artem Spector, Ranit Aharonov, Yoav Katz and Noam Slonim.

The evaluation code was written by [Matan Orbach](https://github.com/matanor) and Artem Spector.

[issues]: https://github.com/IBM/yaso-tsa/issues/new
