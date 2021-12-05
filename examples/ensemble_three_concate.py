# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
from allennlp_models import pretrained
# load pretrained lstm
predictor = pretrained.load_predictor(
    'tagging-fine-grained-crf-tagger', cuda_device=0)

import sklearn

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_VEC = 'glove/glove.840B.300d.txt'
# PATH_TO_VEC = 'fasttext/crawl-300d-2M.vec'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        # lstm embedding
        input = {"sentence": ' '.join(sent)}
        with predictor.capture_model_internals('encoder') as internals:
            outputs = predictor.predict_json(input)
            for k, v in internals.items():
                embedding = v['output']

        sentvec_lstm = np.array(embedding)[0]
        sentvec_lstm = np.mean(sentvec_lstm, 0)
        sentvec_roberta = model.encode(sent)
        sentvec_roberta = np.mean(sentvec_roberta, 0)

        sentvec_mpnet = model2.encode(sent)
        sentvec_mpnet = np.mean(sentvec_mpnet, 0)

        sentvec_glove = []
        for word in sent:
            if word in params.word_vec:
                sentvec_glove.append(params.word_vec[word])
        if not sentvec_glove:
            vec = np.zeros(params.wvec_dim)
            sentvec_glove.append(vec)
        sentvec_glove = np.mean(sentvec_glove, 0)
        
        concatenated = np.concatenate((sentvec_roberta, sentvec_lstm, sentvec_glove))
        # concatenated = np.concatenate((sentvec_roberta, []))
        # concatenated = np.concatenate((sentvec_glove, sentvec_roberta))

        # print("roberta: ", np.sum(sentvec_roberta**2))
        # print("glove: ", np.sum(sentvec_glove**2))
        # norm = np.linalg.norm(concatenated)
        # print("normalized: ", np.sum((concatenated/norm)**2))
        
        # concatenated /= norm
        embeddings.append(concatenated)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                  'tenacity': 3, 'epoch_size': 2}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 1, 'dropout': 0.1, 'max_epoch': 200}

# Set up logger

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO,
handlers=[
    logging.FileHandler("benchmark-result/three-concate-ensemble-copy.log"),
    logging.StreamHandler()
])

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
    #                   'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
    #                   'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['BigramShift', 'CoordinationInversion']
                      # 'Length', 'WordContent', 'Depth', 'TopConstituents',
                      # 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      # 'OddManOut', '']
    # transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    # transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC']
    results = se.eval(transfer_tasks)
    print(results)
