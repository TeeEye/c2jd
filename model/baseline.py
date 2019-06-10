from torch import nn
import torch
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, hidden_size=300, embeds_dim=300):
        super(Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        num_word = 48000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.lstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, *inputs):
        # batch_size * seq_len
        sent1, sent2 = inputs[0], inputs[1]

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.embeds(sent1)
        x2 = self.embeds(sent2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm(x1)
        o2, _ = self.lstm(x2)

        # Classifier
        x = (o1[-1, 0] + o2[-1, 0]) / 2
        x = self.fc(x)
        return F.sigmoid(x)


if __name__ == '__main__':
    b = Baseline(hidden_size=300, embeds_dim=300)
    print(b)
