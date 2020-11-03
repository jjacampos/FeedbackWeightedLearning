# Improving Conversational Question Answering Systems after Deployment using Feedback Weighted Learning

In this repository you can find the code developed for Feedback Weighted Learning (FWL) in the document classification scenario. For the Conversational QA scenario we substituted the standard Cross Entropy loss at the QA script on Huggingface with FWL. 

# Requirements

All the requirements are specified in the requirements.txt file and can be installed using the following command:

```
pip install -r requirements.txt
```

# Dataset splits

In the data folder you can find the indexes for the new *training* and *deployment* splits used in this work. The new data splits are created from the original *training* split and are divided in the following way:

* doc_class_splits folder contains the line number of the examples in the original training file of the DBPedia Classes dataset: https://www.kaggle.com/danofer/dbpedia-classes. The *development* and *test* splits are kept unchanged. 

* QuAC_splits folder contains the id of the examples in the original training file of the QuAC dataset: http://quac.ai/. The *development* split is kept unchanged and the test split is not publicly available. 

# Reference

If you use this code, please cite us:

```
@inproceedings{campos2020improving,
  title = {{Improving Conversational Question Answering Systems after Deployment using Feedback Weighted Learning}},
  author = {Campos, Jon Ander and Cho, Kyunghyn and Otegi, Arantxa and Soroa, Aitor and Azkune, Gorka and Agirre, Eneko},
  booktitle = {Proceedings of COLING 2020, the 28th International Conference on Computational Linguistics},
  address = {Online},
  year = {2020},
}
```
