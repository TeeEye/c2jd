from utils.macros import *
import pickle
import numpy as np


class Embedding:
    def __init__(self):
        print('Building tencent embedding...')
        self.word2vec = {}
        with open(EMBEDDING_PATH, 'rb') as f:
            while True:
                try:
                    dic = pickle.load(f)
                    for k, v, in dic.items():
                        if len(k) > 6:
                            continue
                        self.word2vec[k] = v
                    del dic
                except EOFError:
                    break
        print('Embedding initiated!')

    def sentence2vec(self, sentence):
        result = []
        for word in sentence:
            if len(result) >= PAD_SIZE:
                break
            if word not in self.word2vec:
                continue
            result.append(self.word2vec[word])
        return np.array(result)
