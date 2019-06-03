"""
    该模块定义了词典，主要作用是 word2idx
"""
from utils.macros import *
import sys
sys.path.append('..')


class Voc:
    """
        词典类，负责将词语字符串转化为索引
        对外接口：
        add_word(word)
        trim()
    """
    def __init__(self):
        self._trimmed = False
        self._reset()

    def _reset(self):
        self._word2idx = {PAD_STR: PAD_TOKEN, SOS_STR: SOS_TOKEN,
                          EOS_STR: EOS_TOKEN, UNK_STR: UNK_TOKEN}
        self._word_count = {}
        self._num_words = 4  # SOS, EOF, PAD, UNK

    def add_word(self, word):
        if word in self._word2idx:
            self._word_count[word] += 1
        else:
            self._word_count[word] = 1
            self._word2idx[word] = self._num_words
            self._num_words += 1

    def trim(self, min_count=MIN_COUNT):
        if self._trimmed:
            return
        self._trimmed = True
        words_kept = []

        for word, count in self._word_count.items():
            if count >= min_count:
                words_kept.append(word)

        if DEBUG_MODE:
            print('Words kept: %d, keeping ratio = %f' %
                  (len(words_kept), len(words_kept)/len(self._word_count)))

        self._reset()

        for word in words_kept:
            self.add_word(word)

    def word2idx(self, word):
        """
        返回输入词的 index
        :param word: 输入词字符串
        :return: 如果词典中存在则返回对应 idx, 否则返回 UNK 的 idx
        """
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx[UNK_STR]
