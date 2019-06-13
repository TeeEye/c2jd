"""
RNN 相关的辅助函数库
"""
import torch.nn as nn
from utils.macros import *


def zero_pad(vecs):
    lens = []
    result = []
    for vec in vecs:
        result.append(torch.from_numpy(vec))
        lens.append(vec.shape[0]-1)
    result = torch.nn.utils.rnn.pad_sequence(result, batch_first=True).to(device)
    lens = torch.tensor(lens).long().to(device)
    return result, lens
