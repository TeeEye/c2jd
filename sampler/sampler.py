"""
从训练集中取数据
"""

from utils.macros import *
import sys
import pickle
import pandas as pd
import random
sys.path.append('..')


class Sampler:
    def __init__(self, batch_size=BATCH_SIZE, train_test_split=TRAIN_TEST_SPLIT,
                 data_size=DATA_SIZE, data_path=TRAIN_PATH):
        seed = random.randint(0, 1000)
        print("Sampler initiating with seed: %d" % seed)
        random.seed(seed)
        _current_data_index = 0  # DATA_PATH 共八个文件, 目前只用第一个文件
        with open(data_path % _current_data_index, 'rb') as data_file:
            data_array = []
            current_len = 0
            while current_len < data_size:
                try:
                    data = (pickle.load(data_file))
                    current_len += len(data)
                    data_array.append(data)
                except EOFError:
                    break
        self.data = pd.concat(data_array)
        self.test = self.data[len(self.data)*train_test_split:]
        self.batch_size = batch_size
        print("Sampler initiated!")

    def next_batch(self):
        start = random.randint(0, len(self.data)-self.batch_size)
        return self.data[start:start+self.batch_size]


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
