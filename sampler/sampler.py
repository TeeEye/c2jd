from utils.macros import *
import pickle
import random
import numpy as np
from utils.rnn import zero_pad


class Sampler:
    """
    取样器, 从 utils.preprocess 生成的 app_train_%d.pkl 中读取数据传递给 solver
    数据源每次读取最多有 1000 行数据, 所以主要逻辑是多次 load, 每次 load 出来的数据
    可以重复使用多次, 通过维护一个 counter 来表示目前的数据的使用数量, 如果大于一定值
    则进行下一次 load. 如果出现 EOF, 则重新读取文件再来一次
    """
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
        print("Sampler initiated!")
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
            cs = row['candidate_summary']
            jd = row['job_description']
            if cs is None or jd is None or \
                    cs.shape[0] < MIN_AVAILABLE_SENTENCE_LEN or jd.shape[0] < MIN_AVAILABLE_SENTENCE_LEN:
                continue
            summary.append(cs)
            description.append(jd)
            label.append(row['label'])
        self.summary, self.len1 = zero_pad(summary)
        self.description, self.len2 = zero_pad(description)
        self.label = torch.from_numpy(np.array(label).reshape(-1, 1)).float().to(device)

    def next_batch(self):
        """
        随机返回下一个 batch
        counter 记录了该批数据的使用数量, 如果大于阈值则使用下批数据
        :return: 下一个 batch 的两个字段和对应长度
        """
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

    def test_set(self):
        self.load_data()
        return self.summary, self.description, self.label, self.len1, self.len2


if __name__ == '__main__':
    s = Sampler(batch_size=16)
    print(s.next_batch())
