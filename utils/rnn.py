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
        lens.append(vec.shape[0])
    result = torch.nn.utils.rnn.pad_sequence(result, batch_first=True).float().to(device)
    lens = torch.tensor(lens).long().to(device)
    return result, lens


def run_rnn(rnn, data, lens):
    sorted_lens, indices = torch.sort(lens, descending=True)
    _, desorted_indices = torch.sort(indices)
    data = data.index_select(0, indices)
    packed = nn.utils.rnn.pack_padded_sequence(data, sorted_lens, batch_first=True)
    res, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True, total_length=data.shape[1])
    res = padded.index_select(0, desorted_indices).contiguous()
    return batch_select(res, lens-1)


def batch_select(tensor, index):
    return tensor.gather(1, index.view(-1, 1, 1).expand(tensor.size(0), 1, tensor.size(2))).squeeze(1)

