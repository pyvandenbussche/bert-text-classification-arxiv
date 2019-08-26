# bert-text-classificaiton-arxiv


[![Documentation Status](https://img.shields.io/badge/Blog-link_to_the_post-brightgreen.svg)](http://pyvandenbussche.info/2019/ai-or-not-ai-classifying-arxiv-articles-with-bert/)

AI or not AI? Classifying ArXiv articles with BERT

## Installation

### Prerequisites

* Python ≥ 3.6

### Provision a Virtual Environment

Create and activate a virtual environment (conda)

```
conda create --name py36_bert-arxiv python=3.6
source activate py36_bert-arxiv
```

If `pip` is configured in your conda environment, 
install dependencies from within the project root directory
```
pip install -r requirements.txt
``` 

## Get ArXiv dataset

The dataset used in this repository should be [downloaded from Kaggle](https://www.kaggle.com/neelshah18/arxivdataset)

Create a folder `data` from within the project root directory.
Place the downloaded file `arxivData.json` in the `data` folder.

## Feature Extraction code

Now that the environment is setup and the dataset is available, you can run the code using the following command:
```bash
python feature_extraction.py 
```
This will by default use the `arxivData.json` file as input and generate in the same `data` folder the X,y training and test files:

## model training
Use the jupyter notebook `run_model_keras` to train the model. 
This is easier to visualise the results we get.
