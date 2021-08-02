This package contains the YASO evaluation dataset for targeted sentiment analysis (TSA).

Some of the sentences annotated in YASO are taken from other datasets that cannot be re-distributed in clear text. 
That's why for some records the  file **yaso_hidden.json** includes an ID instead of plain text. 

To retrieve the original texts:
 
 (1) Download the required resources from the other datasets, as described below.  
 (2) Run a script that restores the original texts.

The resulting **yaso.json** file will contain the annotated sentences with their targets and sentiments.

The **yaso.json** file contains a JSON array of examples, where each example includes one annotated text and a list of targets and their sentiments identified within that text. 
Each target has its text, location within the sentence, and sentiment. 
The sentiment value can be **positive**, **negative**, **mixed**, or **none**.

An example of one annotated sentence:
```json
  {
    "text": "Great food but the service was dreadful !",
    "targets": [
      {
        "text": "food",
        "location": {
          "begin": 6,
          "end": 10
        },
        "sentiment": "positive"
      },
      {
        "text": "service",
        "location": {
          "begin": 19,
          "end": 26
        },
        "sentiment": "negative"
      }
    ]
  }
```

License
---
This dataset is released under Community Data License Agreement – Sharing, Version 1.0 (https://cdla.dev/sharing-1-0/)

Downloading dataset resources
---

### Amazon
Download the English test file from Multilingual Amazon Reviews Corpus.
Access the data as described here: https://docs.opendata.aws/amazon-reviews-ml/readme.html#access 

Download the test file https://amazon-reviews-ml.s3-us-west-2.amazonaws.com/json/test/dataset_en_test.json
and use its path as value for the **--amazon** argument:
```commandline
--amazon ~/Downloads/dataset_en_test.json
```

### SST
Download the data set from  https://nlp.stanford.edu/sentiment/index.html 
("Main zip file" link on the right side: http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) and unzip it.

Use the directory path as value for the **--sst** argument:
```commandline
--sst ~/Downloads/stanfordSentimentTreebank
```

### Opinosis
Download from the repository https://github.com/kavgan/opinosis-summarization the file `OpinosisDataset1.0_0.zip` and unzip it.

Use the path of the **topics** directory as value for argument **--opinosis**:
```commandline
--opinosis ~/Downloads/OpinosisDataset1.0/topics
```

### SemEval14
Download SemEval14 ABSA Test Data:
Access MetaShare at http://metashare.elda.org/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/, 
login, download and unzip the data file.

Use the path of the directory as value for the **--semeval** argument :
```commandline
--semeval ~/Downloads/ABSA_Gold_TestData
```

Run the restore script
-----------------
Requirements:
* Python 3
* pandas
* nltk


Run `restore_texts.py` with the arguments described above.
All arguments are optional, skipping an argument for a source will not restore its sentences. 

```commandline
python restore_texts.py \
    --amazon ~/Downloads/dataset_en_test.json \
    --sst ~/Downloads/stanfordSentimentTreebank \
    --opinosis ~/Downloads/OpinosisDataset1.0/topics \
    --semeval ~/Downloads/ABSA_Gold_TestData 
```

The dataset with the restored sentences is saved to  `yaso.json`.

