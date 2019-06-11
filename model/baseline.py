from torch import nn
import torch


def batch_select(tensor, index):
    return tensor.gather(1, index.view(-1, 1, 1).expand(tensor.size(0), 1, tensor.size(2))).squeeze(1)


def RunRnn(rnn, inputs, seq_lengths):
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
    _, desorted_indices = torch.sort(indices, descending=False)
    inputs = inputs.index_select(0, indices)
    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
    res, _ = rnn(packed_inputs)
    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True, total_length=inputs.shape[1])
    return padded_res.index_select(0, desorted_indices).contiguous()


class Baseline(nn.Module):
    def __init__(self, hidden_size=300, embeds_dim=200):
        super(Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        self.lstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 4, 1)

    def forward(self, *inputs):
        # batch_size * seq_len
        x1, x2 = inputs[0], inputs[1]
        len1, len2 = inputs[2], inputs[3]
        # embeds: batch_size * seq_len => batch_size * seq_len * dim

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1 = RunRnn(self.lstm, x1, len1)
        o2 = RunRnn(self.lstm, x2, len2)

        o1 = batch_select(o1, len1)
        o2 = batch_select(o2, len2)

        # Classifier
        x = torch.cat([o1, o2], 1)
        x = self.fc(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    b = Baseline(hidden_size=300, embeds_dim=300)
    print(b)
