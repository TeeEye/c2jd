"""
    该模块定义了词典，主要作用是 sentence2idxs
    用法：
    voc = Voc() # 该步直接生成可用的词典
    print(voc.num_words)
    ...
    sentences = [s1, s2, ..., s3]
    idxs = voc.sentence_batch_2_idxs(sentences)

"""
from utils.macros import *
import numpy as np
import jieba
import sys
import os
sys.path.append('..')


class Voc:
    """
        词典类，负责将词语字符串转化为索引
        对外接口：
        add_word(word)
        trim()
        word2idx(word)
        sentence2idxs(sentence)
    """
    def __init__(self):
        self._word2idx = {UNK_STR: UNK_TOKEN}
        self.num_words = 1  # UNK
        assert os.path.exists(VOCABULARY_PATH)
        print('Start building vocabulary...')
        with open(VOCABULARY_PATH) as vocabulary_text:
            for word in vocabulary_text:
                self.add_sentence(word)
        print('Vocabulary initiated!')

    def add_word(self, word):
        word = word.lower()
        if word in self._word2idx:
            return
        self._word2idx[word] = self.num_words
        self.num_words += 1

    def add_sentence(self, sentence):
        for word in jieba.cut(sentence):
            self.add_word(word)

    def word2idx(self, word):
        """
        返回输入词的 index
        :param word: 输入词字符串
        :return: 如果词典中存在则返回对应 idx, 否则返回 UNK 的 idx
        """
        word = word.lower()
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return UNK_TOKEN

    def sentence2idxs(self, sentence):
        result = []
        for word in sentence:
            idx = self.word2idx(word)
            print(word, idx)
            if idx == UNK_TOKEN:
                continue
            result.append(idx)
        return np.array(result)


if __name__ == '__main__':
    voc = Voc()
    print(voc.num_words)
    while True:
        s = input('input a sentence...: ')
        s = jieba.cut(s)
        print(voc.sentence2idxs(s))
