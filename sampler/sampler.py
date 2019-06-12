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
    def __init__(self, batch_size=BATCH_SIZE, data_path=TRAIN_PATH):
        seed = random.randint(0, 1000)
        print("Sampler initiating with seed: %d" % seed)
        random.seed(seed)
        self.data_path = data_path
        self.batch_size = batch_size
        self.current_data_index = 0  # DATA_PATH 共八个文件, 目前只用第一个文件
        self.data_file = open(data_path % self.current_data_index, 'rb')
        self.summary = None
        self.len1 = None
        self.description = None
        self.len2 = None
        self.label = None
        self.counter = 0
        self.load_data()

    def load_data(self):
        print('Sampler load data')
        try:
            data = pickle.load(self.data_file)
        except EOFError:
            self.data_file.close()
            self.data_file = open(self.data_path % self.current_data_index, 'rb')
            data = pickle.load(self.data_file)

        # 将 DataFrame 转化为 np.array 且 batch_size 为第二维度
        summary = []
        description = []
        label = []
        for _, row in data.iterrows():
            summary.append(row['candidate_summary'])
            description.append(row['job_description'])
            label.append(row['label'])
        self.summary, self.len1 = self.zero_pad(summary)
        self.description, self.len2 = self.zero_pad(description)
        self.label = np.array(label)
        print("Sampler initiated!")

    def zero_pad(self, inputs):
        result = np.zeros((len(inputs), PAD_SIZE, EMBED_DIM))
        inputs_len = []
        for index, input in enumerate(inputs):
            input_len = min(PAD_SIZE, len(input))
            for i in range(input_len):
                result[index, i] = input[i]
            inputs_len.append(input_len-1)
        return result, inputs_len

    def next_batch(self):
        self.counter += self.batch_size
        if self.counter > len(self.summary) * TRAIN_DATA_REUSE_TIMES:
            self.load_data()
            self.counter = self.batch_size
        start = random.randint(0, len(self.summary)-self.batch_size)
        return self.summary[start:start+self.batch_size], \
            self.description[start:start+self.batch_size], \
            self.label[start:start+self.batch_size], \
            self.len1[start:start+self.batch_size], \
            self.len2[start:start+self.batch_size]


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
