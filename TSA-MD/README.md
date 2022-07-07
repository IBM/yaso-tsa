# TSA-MD: A Targeted Sentiment Analysis Training Dataset over Open-Domain Reviews

This folder contains the Targeted Sentiment Analysis Multi-Domain (TSA-MD) dataset.

This dataset is described in the paper [Multi-Domain Targeted Sentiment Analysis (Toledo-Ronen et al. 2022)](https://arxiv.org/abs/2205.03804), published in NAACL 2022.

Overall, the dataset contains 952 sentences of reviews from multiple domains. 

## Collection

Data collection started with reviews written by crowd annotators in a given domain, on a topic of their choice. 
The reviews were then annotated for TSA by asking annotators to mark all sentiment-bearing targets in
each sentence. 
This step is similar to the candidates
annotation phase described in [Orbach et al. (2021)](https://aclanthology.org/2021.emnlp-main.721/). 
However, unlike that work, the detected candidates were not passed through another verification step, to reduce costs.
This results in noisier data, unfit for evaluation purposes, yet a manual examination has shown it is of sufficient quality for training. 
Experiments showing the merit of using this data in training are reported in our paper [Multi-Domain Targeted Sentiment Analysis (Toledo-Ronen et al. 2022)](https://arxiv.org/abs/2205.03804).

## Split
The dataset contains a training set and a development set, created by randomly shuffling all the available sentences, and selecting 80% for the training set (761 sentences) and 20% for development (191 sentences).  
The data is **not intended for use as an evaluation set**, as it is noisy.