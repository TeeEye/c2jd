"""
从训练集中取数据
"""

import sys
sys.path.append('..')
from utils.macros import *
import pickle
import random
import numpy as np


class Sampler:
    def __init__(self, batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT,
                 data_path=TRAIN_PATH):
        seed = random.randint(0, 1000)
        print("Sampler initiating with seed: %d" % seed)
        random.seed(seed)
        self.batch_size = batch_size
        _current_data_index = 0  # DATA_PATH 共八个文件, 目前只用第一个文件
        with open(data_path % _current_data_index, 'rb') as data_file:
            self.data = pickle.load(data_file)
        test = self.data[int(len(self.data)*train_test_split):]
        self.test_x = np.zeros((self.batch_size, 2, MAX_SENTENCE_LEN))
        self.test_y = []
        for idx, row in test.iterrows():
            for i in range(min(MAX_SENTENCE_LEN, len(row['candidate_summary']))):
                self.test_x[idx, 0, i] = row['candidate_summary'][i]
            for i in range(min(MAX_SENTENCE_LEN, len(row['job_description']))):
                self.test_x[idx, 1, i] = row['candidate_summary'][i]
            self.test_y.append(row['label'])
        self.test_y = np.array(self.test_y)
        print("Sampler initiated!")

    def next_batch(self):
        start = random.randint(0, len(self.data)-self.batch_size)
        batch = self.data[start:start+self.batch_size]
        x = np.zeros((self.batch_size, 2, MAX_SENTENCE_LEN))
        y = []
        for idx, row in batch.iterrows():
            for i in range(min(MAX_SENTENCE_LEN, len(row['candidate_summary']))):
                x[idx, 0, i] = row['candidate_summary'][i]
            for i in range(min(MAX_SENTENCE_LEN, len(row['job_description']))):
                x[idx, 1, i] = row['candidate_summary'][i]
            y.append(row['label'])
        return x, np.array(y)

    def test(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
