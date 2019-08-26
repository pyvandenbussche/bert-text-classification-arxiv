'''
Data preparation and feature extraction of arXiv dataset available at
https://www.kaggle.com/neelshah18/arxivdataset
'''

import argparse
import beautifultable as bt
from bert_serving.client import BertClient
import collections
from gensim.models import Word2Vec
import pandas as pd
import networkx as nx
import numpy as np
import re
from sklearn.model_selection import train_test_split
from src import node2vec
import time
import os

LABELS_FP = "data/labels.tsv"
TITLE_FEATURES_FP = "data/titles_features.tsv"
PARAGRAPHS_FEATURES_FP = "data/paragraphs_features.tsv"
NODE_FEATURES_FP = "data/node_features.tsv"
TITLES_FP = "data/titles.tsv"
IDs_FP = "data/ids.tsv"
RANDOM_SEED = 1234


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="http://pyvandenbussche.info/2019/ai-or-not-ai-classifying-arxiv-articles-with-bert/")

    parser.add_argument('--input', nargs='?', default='data/arxivData.json', help='Input Arxiv json file')
    return parser.parse_args()


def load_data(input_file, tag_of_interest):
    '''
    Load arXiv data
    :param input_file: path to arxiv file
    '''

    # read the arxiv input json file
    df = pd.read_json(input_file, orient='records')

    # flatten author list names.
    # this is not the most elegant but is made to handle the variation in single/double quotes for name values:
    # "author": "[{'name': 'Luciano Serafini'}, {'name': \"Artur d'Avila Garcez\"}]",
    df['author_list'] = df['author'].apply(lambda author_str: [x.strip()[10:-2] for x in author_str[1:-1].split(",")])

     # flatten tags list
    def flatten_tags(tag_str):
        tags = tag_str[1:-1].split("{'term': '")
        tags = list(filter(None, [tag.strip()[:tag.find("'")] for tag in tags]))
        return tags
    df['tags_list'] = df['tag'].apply(flatten_tags)

    unique_lists_tags = df['tags_list'].values
    df['Y'] = [int((tag_of_interest in tags) == True) for tags in unique_lists_tags]
    print("\t- found {} articles with tag {}".format(df['Y'].sum(), tag_of_interest))

    return df

def get_titles(df):

    titles = df["title"].values.tolist()
    titles = [title.replace('\n ', '').replace('\r', '').lower() for title in titles]
    return np.array(titles)


def get_sentences_embedding(sentences):
    '''
    Query bert server and get back sentence embeddings for each article's title
    :param titles: articles titles
    :return: np array of features
    '''
    bc = BertClient()
    X = bc.encode(sentences)
    return X


def main(args):
    '''
    Pipeline for representational learning for all nodes in the ArXiv graph.
    '''
    print("Loading Arxiv Data")
    # load data:
    df = load_data(input_file=args.input, tag_of_interest="cs.CV")
    # df = load_data(input_file=args.input, tag_of_interest="cs.AI")


    print("Saving labels Y")
    np.savetxt(LABELS_FP, df["Y"].values, fmt='%i', delimiter='\t')

    print("Saving titles and ids")
    if os.path.isfile(TITLES_FP) & os.path.isfile(IDs_FP):
        titles = get_titles(df)
        print("\t- Files already exist. Will reuse them")
    else:
        titles = get_titles(df)
        np.savetxt(TITLES_FP, titles, fmt='%s', delimiter='\t')
        np.savetxt(IDs_FP, df["id"].values, fmt='%s', delimiter='\t')

    print("Computing titles embeddings")
    if os.path.isfile(TITLE_FEATURES_FP):
        print("\t- File already exists. Will reuse it")
        title_embed = np.loadtxt(TITLE_FEATURES_FP, delimiter="\t")
    else:
        title_embed = get_sentences_embedding(titles)
        # print(title_embed[0])
        print("\t- Saving title features to file")
        np.savetxt(TITLE_FEATURES_FP, title_embed, delimiter='\t')

    print("Computing paragraphs embeddings")
    if os.path.isfile(PARAGRAPHS_FEATURES_FP):
        print("\t- File already exists. Will reuse it")
    else:
        with open(PARAGRAPHS_FEATURES_FP, 'w') as f:
            for index, row in df.iterrows():
                # Split Paragraph on basis of '.' or ? or !.
                sentences = re.split(r"\.|\?|\!", row["summary"])
                sentences = [sentence.replace('\n ', '').replace('\r', '').strip() for sentence in sentences]
                sentences = list(filter(None, sentences))

                sent_embed = get_sentences_embedding(sentences)
                par_embed = np.average(sent_embed, axis=0)
                np.savetxt(f, par_embed[None], delimiter='\t')
                if index % 100 == 0:
                    print("\t {}/{} paragraphs processed".format(index, len(df.index)))
    par_embed = np.loadtxt(PARAGRAPHS_FEATURES_FP, delimiter="\t")
    print("Computing node embeddings")

    print("Split in train/test sets")
    # get index for train test elements
    y = np.array(df["Y"].values)
    print("y shape: {}".format(y.shape))
    Idx = np.array(range(len(df.index)))
    print("Idx shape: {}".format(Idx.shape))

    train_idx, test_idx, y_train, y_test = train_test_split(Idx, y, random_state=RANDOM_SEED, stratify=y, test_size=.2)

    # X_train = title_embed[train_idx]
    X_train = np.concatenate((title_embed[train_idx], par_embed[train_idx]), axis=1)
    np.savetxt("data/X_train.tsv", X_train, delimiter='\t')
    print("X_train shape: {}".format(X_train.shape))
    np.savetxt("data/y_train.tsv", y_train, delimiter='\t')
    print("y_train shape: {}".format(y_train.shape))
    np.savetxt("data/title_train.tsv", titles[train_idx], fmt="%s", delimiter='\t')

    # X_test = title_embed[test_idx]
    X_test = np.concatenate((title_embed[test_idx], par_embed[test_idx]), axis=1)
    np.savetxt("data/X_test.tsv", X_test, delimiter='\t')
    print("X_test shape: {}".format(X_test.shape))
    np.savetxt("data/y_test.tsv", y_test, delimiter='\t')
    print("y_test shape: {}".format(y_test.shape))
    np.savetxt("data/title_test.tsv", titles[test_idx], fmt="%s", delimiter='\t')




if __name__ == "__main__":
    args = parse_args()
    main(args)
