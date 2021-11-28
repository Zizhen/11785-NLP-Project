# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
MRPC : Microsoft Research Paraphrase (detection) Corpus
'''
from __future__ import absolute_import, division, unicode_literals

import os
import logging
import numpy as np
import io

from senteval.tools.validation import KFoldClassifier

from sklearn.metrics import f1_score, accuracy_score


class MRPCEval(object):
    def __init__(self, task_path, seed=1111):
        logging.info('***** Transfer task : MRPC *****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path,
                              'msr_paraphrase_train.txt'))
        test = self.loadFile(os.path.join(task_path,
                             'msr_paraphrase_test.txt'))
        self.mrpc_data = {'train': train, 'test': test}

    def do_prepare(self, params, prepare):
        # TODO : Should we separate samples in "train, test"?
        samples = self.mrpc_data['train']['X_A'] + \
                  self.mrpc_data['train']['X_B'] + \
                  self.mrpc_data['test']['X_A'] + self.mrpc_data['test']['X_B']
        return prepare(params, samples)

    def loadFile(self, fpath):
        mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                mrpc_data['X_A'].append(text[3].split())
                mrpc_data['X_B'].append(text[4].split())
                mrpc_data['y'].append(text[0])

        mrpc_data['X_A'] = mrpc_data['X_A'][1:]
        mrpc_data['X_B'] = mrpc_data['X_B'][1:]
        mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]
        return mrpc_data

    def run(self, params, batcher):
        mrpc_embed = {'train': {}, 'test': {}}

        for key in self.mrpc_data:
            logging.info('Computing embedding for {0}'.format(key))
            # Sort to reduce padding
            text_data = {}
            sorted_corpus = sorted(zip(self.mrpc_data[key]['X_A'],
                                       self.mrpc_data[key]['X_B'],
                                       self.mrpc_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            text_data['A'] = [x for (x, y, z) in sorted_corpus]
            text_data['B'] = [y for (x, y, z) in sorted_corpus]
            text_data['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['A', 'B']:
                mrpc_embed[key][txt_type] = []
                for ii in range(0, len(text_data['y']), params.batch_size):
                    batch = text_data[txt_type][ii:ii + params.batch_size]
                    embeddings = batcher(params, batch)
                    mrpc_embed[key][txt_type].append(embeddings)
                mrpc_embed[key][txt_type] = np.vstack(mrpc_embed[key][txt_type])
            mrpc_embed[key]['y'] = np.array(text_data['y'])
            logging.info('Computed {0} embeddings'.format(key))

        print(mrpc_embed['train']['A'].shape)
        # clf 1 roberta
        # Train
        trainA_1 = mrpc_embed['train']['A'][:, :768]
        trainB_1 = mrpc_embed['train']['B'][:, :768]
        trainF_1 = np.c_[np.abs(trainA_1 - trainB_1), trainA_1 * trainB_1]
        trainY_1 = mrpc_embed['train']['y']
        # Test
        testA_1 = mrpc_embed['test']['A'][:, :768]
        testB_1 = mrpc_embed['test']['B'][:, :768]
        testF_1 = np.c_[np.abs(testA_1 - testB_1), testA_1 * testB_1]
        testY_1 = mrpc_embed['test']['y']

        config_1 = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': {'nhid': 1000, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 1, 'dropout': 0.1, 'max_epoch': 200},
                  'nhid': 1000, 'kfold': params.kfold}
        clf_1 = KFoldClassifier(train={'X': trainF_1, 'y': trainY_1},
                              test={'X': testF_1, 'y': testY_1}, config=config_1)

        devacc_1, testacc_1, yhat_1 = clf_1.run()
        print('Dev acc 1: {0} Test acc 1: {1} for MRPC.\n'
                      .format(devacc_1, testacc_1))
        
        # clf 2 mpnet
        # Train
        trainA_2 = mrpc_embed['train']['A'][:, 768:1536]
        trainB_2 = mrpc_embed['train']['B'][:, 768:1536]
        trainF_2 = np.c_[np.abs(trainA_2 - trainB_2), trainA_2 * trainB_2]
        trainY_2 = mrpc_embed['train']['y']
        # Test
        testA_2 = mrpc_embed['test']['A'][:, 768:1536]
        testB_2 = mrpc_embed['test']['B'][:, 768:1536]
        testF_2 = np.c_[np.abs(testA_2 - testB_2), testA_2 * testB_2]
        testY_2 = mrpc_embed['test']['y']

        config_2 = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': {'nhid': 1000, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 1, 'dropout': 0.1, 'max_epoch': 200},
                  'nhid': 1000, 'kfold': params.kfold}
        clf_2 = KFoldClassifier(train={'X': trainF_2, 'y': trainY_2},
                              test={'X': testF_2, 'y': testY_2}, config=config_2)

        devacc_2, testacc_2, yhat_2 = clf_2.run()
        print('Dev acc 2: {0} Test acc 2: {1} for MRPC.\n'
                      .format(devacc_2, testacc_2))

        # clf 3 glove
        # Train
        trainA_3 = mrpc_embed['train']['A'][:, 1536:]
        trainB_3 = mrpc_embed['train']['B'][:, 1536:]
        trainF_3 = np.c_[np.abs(trainA_3 - trainB_3), trainA_3 * trainB_3]
        trainY_3 = mrpc_embed['train']['y']
        # Test
        testA_3 = mrpc_embed['test']['A'][:, 1536:]
        testB_3 = mrpc_embed['test']['B'][:, 1536:]
        testF_3 = np.c_[np.abs(testA_3 - testB_3), testA_3 * testB_3]
        testY_3 = mrpc_embed['test']['y']

        config_3 = {'nclasses': 2, 'seed': self.seed,
                  'usepytorch': params.usepytorch,
                  'classifier': {'nhid': 500, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 1, 'dropout': 0.1, 'max_epoch': 200},
                  'nhid': 600, 'kfold': params.kfold}
        clf_3 = KFoldClassifier(train={'X': trainF_3, 'y': trainY_3},
                              test={'X': testF_3, 'y': testY_3}, config=config_3)

        devacc_3, testacc_3, yhat_3 = clf_3.run()
        print('Dev acc 3: {0} Test acc 3: {1} for MRPC.\n'
                      .format(devacc_3, testacc_3))

        yhat = []
        for i in range(len(yhat_1)):
            summ = yhat_1[i] + yhat_2[i] + yhat_3[i]
            if summ <= 1:
                yhat.append(0)
            else:
                yhat.append(1)
        testf1 = round(100*f1_score(testY_1, yhat), 2)
        testacc = round(100*accuracy_score(testY_1, yhat), 2)
        logging.debug('Dev acc : {0} Test acc {1}; Test F1 {2} for MRPC.\n'
                      .format(0, testacc, testf1))
        return {'devacc': 0, 'acc': testacc, 'f1': testf1,
                'ndev': len(trainA_3), 'ntest': len(testA_3)}
