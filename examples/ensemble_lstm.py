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
from allennlp_models import pretrained

for k, v in logging.root.manager.loggerDict.items():
    logging.getLogger(k).disabled = True

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


# load transformer
model = SentenceTransformer(
    'sentence-transformers/paraphrase-distilroberta-base-v2')

# load pretrained lstm
predictor = pretrained.load_predictor(
    'tagging-fine-grained-crf-tagger', cuda_device=0)


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

        # roberta embedding
        sentvec_roberta = model.encode(sent)
        sentvec_roberta = np.mean(sentvec_roberta, 0)

        # glove embedding
        sentvec_glove = []
        for word in sent:
            if word in params.word_vec:
                sentvec_glove.append(params.word_vec[word])
        if not sentvec_glove:
            vec = np.zeros(params.wvec_dim)
            sentvec_glove.append(vec)
        sentvec_glove = np.mean(sentvec_glove, 0)

        # concat
        concatenated = np.concatenate(
            (sentvec_roberta, sentvec_glove, sentvec_lstm))

        embeddings.append(concatenated)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                  'tenacity': 3, 'epoch_size': 2}

params_senteval['classifier'] = {'nhid': 600, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG,
                    handlers=[
                        logging.FileHandler("benchmark-result/test.log"),
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
    # transfer_tasks = ['STSBenchmark',
    #                   'Length', 'WordContent', 'Depth', 'TopConstituents',
    #                   'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
    #                   'OddManOut', 'CoordinationInversion']
    # transfer_tasks = ['SNLI']
    transfer_tasks = ['MRPC']
    results = se.eval(transfer_tasks)
    print(results)
