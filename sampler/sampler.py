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
        _current_data_index = 0  # DATA_PATH 共八个文件, 目前只用第一个文件
        with open(data_path % _current_data_index, 'rb') as data_file:
            self.data = pickle.load(data_file)
        test = self.data[int(len(self.data)*train_test_split):]
        self.test_x = []
        self.test_y = []
        for _, row in test.iterrows():
            self.test_x.append(np.array([np.array(row['candidate_summary']), np.array(row['job_description'])]))
            self.test_y.append(row['label'])

        self.batch_size = batch_size
        print("Sampler initiated!")

    def next_batch(self):
        start = random.randint(0, len(self.data)-self.batch_size)
        batch = self.data[start:start+self.batch_size]
        x = []
        y = []
        for _, row in batch.iterrows():
            x.append(np.array([np.array(row['candidate_summary']), np.array(row['job_description'])]))
            y.append(row['label'])
        return np.array(x), np.array(y)

    def test(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
