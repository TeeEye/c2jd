"""
从训练集中取数据
"""

import sys
sys.path.append('..')
from utils.macros import *
import pickle
import random
import numpy as np
from itertools import zip_longest


class Sampler:
    def __init__(self, batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT,
                 data_path=TRAIN_PATH):
        seed = random.randint(0, 1000)
        print("Sampler initiating with seed: %d" % seed)
        random.seed(seed)
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        _current_data_index = 0  # DATA_PATH 共八个文件, 目前只用第一个文件
        with open(data_path % _current_data_index, 'rb') as data_file:
            data = pickle.load(data_file)

        # 将 DataFrame 转化为 np.array 且 batch_size 为第二维度
        summary = []
        description = []
        label = []
        for _, row in data.iterrows():
            summary.append(row['candidate_summary'])
            description.append(row['job_description'])
            label.append(row['label'])
        self.summary = self.zero_pad(summary)
        self.description = self.zero_pad(description)
        self.label = label
        test_start = int(len(self.summary) * train_test_split)
        self.test_s = self.summary[test_start:]
        self.test_d = self.description[test_start:]
        self.test_l = self.label[test_start:]
        print("Sampler initiated!")

    def zero_pad(self, inputs):
        zipped = zip_longest(*inputs, fillvalue=0)
        return np.concatenate([np.array(z).reshape(1, -1) for z in zipped])

    def next_batch(self):
        start = random.randint(0, int(len(self.summary) * self.train_test_split)-self.batch_size)
        return self.summary[:, start:start+self.batch_size], \
            self.description[:, start:start+self.batch_size], \
            self.label[:, start:start+self.batch_size]

    def test(self):
        return self.test_s, self.test_d, self.test_l


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
