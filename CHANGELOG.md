# Changelog

All notable changes to this project will be documented in this file.

## 2021-Aug-02

Initial version

## 2022-Jul-07

Added the [TSA-MD dataset](./TSA-MD/README.md).

## <a name="added_domain_labels_to_yaso"></a>2022-Jul-18

Added the `fine-grained-domain` and `domain` fields to the texts in the
[YASO json file](./yaso_tsa/data/yaso_hidden.json).
These values were annotated in our work on [Multi-Domain Targeted Sentiment Analysis (Toledo-Ronen et al. 2022)](https://aclanthology.org/2022.naacl-main.198/), published in NAACL 2022.  

The `fine-grained-domain` values were
produced automatically, when possible, or otherwise
they were manually set by one of the authors.
Since YASO contains annotated reviews from multiple
sources, the assigned `fine-grained domain` label depended on the
source:   
- Reviews from the Stanford Sentiment
Treebank (Socher et al., 2013; Pang and Lee, 2005)
were assigned the `movies` label.
- Reviews from the OPINOSIS source (Ganesan et al., 2010)
were assigned a label of `electronics`, `automotive` or
`hotels`, based on the topic provided in that corpus
for each review. For example, reviews on `transmission_
toyota_camry_2007` were assigned to `automotive`.
- In the YELP source, each review is associated
with a list of business categories. These categories
were used as labels: we manually selected
8 prominent categories as labels, and automatically
matched the reviews to these 8 labels using
the category lists. Reviews matched to multiple
categories were manually examined and assigned
the most fitting label according to the judgement of the authors.
- Texts from the AMAZON source (Keung
et al., 2020) were manually read and labeled.

Finally, the `fine-grained domain` labels were categorized
into a coarse-grained `domain` label: 
- `restaurants` (with 400 sentences)
- `electronics` (412)
- `hotels` (161)
- `automotive` (144)
- `movies` (500)  
- `other` (596)
