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
    parser = argparse.ArgumentParser(description="http://pyvandenbussche.info/2019/ai-or-not-ai-classifying-arxiv-articles-with-bert/.")

    parser.add_argument('--input', nargs='?', default='data/arxivData.json', help='Input Arxiv json file')
    parser.add_argument('--outputdir', nargs='?', default='data/', help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)
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

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]  # convert each vertex id to a string
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    # word_vectors = model.wv
    # del model
    # model.wv.save_word2vec_format(os.path.join(args.outputdir, EMBED_FILE))

    return

def build_kg(df, train_idx):
    '''
    for each row create an edge for each:
    - paper -> author
    - paper -> tag
    '''

    kg = []
    author_set = set()
    tag_set = set()
    article_set = set()
    cpt_hasAuthor = cpt_hasTag = 0

    for index, row in df.iterrows():
        if index in train_idx:
            paper_id = row['id']

            for author in row['author_list']:
                kg.append([paper_id, 'hasAuthor', author])
                author_set.add(author)
                article_set.add(paper_id)
                cpt_hasAuthor += 1
            for tag in row['tags_list']:
                kg.append([paper_id, 'hasTag', tag])
                article_set.add(paper_id)
                tag_set.add(tag)
                cpt_hasTag+=1

    kg = np.asarray(kg)

    # output KG stats
    table = bt.BeautifulTable()
    table.append_row(["# statements", kg.shape[0]])
    table.append_row(["# relation type", 2])
    table.append_row(["   # hasAuthor relation", cpt_hasAuthor])
    table.append_row(["   # hasTag relation", cpt_hasTag])
    table.append_row(["# entities of type Author", len(author_set)])
    table.append_row(["# entities of type Papers", len(article_set)])
    table.append_row(["# entities of type Tag", len(tag_set)])
    table.column_alignments[0] = bt.ALIGN_LEFT
    table.column_alignments[1] = bt.ALIGN_RIGHT
    print(table)

    print("Loading the KG in Networkx")
    # create an id to subject/object label mapping
    set_nodes = set().union(kg[:, 0], kg[:, 2])
    # save label dictionary to file
    node_to_idx = collections.OrderedDict(zip(set_nodes, range(len(set_nodes))))
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    # np.savetxt(os.path.join(args.outputdir, LABEL_FILE), idx_to_node, delimiter="\t", fmt="%s", encoding="utf-8")

    nx_G = nx.DiGraph()
    nx_G.add_nodes_from(range(len(set_nodes)))
    for s, p, o in kg:
        nx_G.add_edge(node_to_idx[s], node_to_idx[o], type=p, weight=1)
    G_undir = nx_G.to_undirected()

    print("Computing transition probabilities and simulating the walks")
    start_time = time.time()
    G = node2vec.Graph(G_undir, False, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    print("Learning the embeddings and writing them to file")
    # print(walks[0])
    walks = [list(map(lambda x: idx_to_node[x], walk)) for walk in walks]  # convert each vertex id to a string
    # print(walks[0])
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    node_vectors = model.wv
    del model


    # get embeddings for paper nodes in train set
    node_paper_embed = []
    for id in df.iloc[train_idx]["id"].values:
        node_paper_embed.append(node_vectors[id])

    node_paper_embed = np.array(node_paper_embed)
    print(node_paper_embed.shape)

    elapsed_time = time.time() - start_time
    print("Node2vec algorithm took: {}".format(time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed_time))))

    return node_paper_embed





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

    node_paper_embed_train = build_kg(df, train_idx)

    # for papers in test set, get 3 closest ones in train set and average their node embedding
    topk = 3
    node_paper_embed_test=[]
    for abstract_emb in par_embed[test_idx]:
        # compute normalized dot product as score
        score = np.sum(abstract_emb * par_embed[train_idx], axis=1) / (np.linalg.norm(par_embed[train_idx], axis=1) * np.linalg.norm(abstract_emb))
        topk_idx = np.argsort(score)[::-1][:topk]

        # average 3 papers node embeddings
        node_paper_embed_test.append( np.average(node_paper_embed_train[topk_idx], axis=0))
    node_paper_embed_test = np.array(node_paper_embed_test)

    X_train = np.concatenate((title_embed[train_idx], par_embed[train_idx], node_paper_embed_train), axis=1)
    np.savetxt("data/X_train.tsv", X_train, delimiter='\t')
    print("X_train shape: {}".format(X_train.shape))
    np.savetxt("data/y_train.tsv", y_train, delimiter='\t')
    print("y_train shape: {}".format(y_train.shape))
    np.savetxt("data/title_train.tsv", titles[train_idx], fmt="%s", delimiter='\t')

    X_test = np.concatenate((title_embed[test_idx], par_embed[test_idx], node_paper_embed_test), axis=1)
    np.savetxt("data/X_test.tsv", X_test, delimiter='\t')
    print("X_test shape: {}".format(X_test.shape))
    np.savetxt("data/y_test.tsv", y_test, delimiter='\t')
    print("y_test shape: {}".format(y_test.shape))
    np.savetxt("data/title_test.tsv", titles[test_idx], fmt="%s", delimiter='\t')




if __name__ == "__main__":
    args = parse_args()
    main(args)
