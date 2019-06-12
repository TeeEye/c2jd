from torch import nn
from utils.macros import *


def batch_select(tensor, index):
    return tensor.gather(1, index.view(-1, 1, 1).expand(tensor.size(0), 1, tensor.size(2))).squeeze(1)


class Baseline(nn.Module):
    def __init__(self, hidden_size=HIDDEN_DIM, embeds_dim=EMBED_DIM):
        super(Baseline, self).__init__()
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        self.lstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 1),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(self.hidden_size * 1, 1)
        )

    def forward(self, *inputs):
        # batch_size * seq_len
        x1, x2 = inputs[0], inputs[1]
        len1, len2 = inputs[2], inputs[3]
        # embeds: batch_size * seq_len => batch_size * seq_len * dim

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm(x1)
        o2, _ = self.lstm(x2)

        o1 = batch_select(o1, len1)
        o2 = batch_select(o2, len2)

        # Classifier
        x = torch.cat([o1, o2], 1)
        x = self.fc(x)
        return torch.sigmoid(x)


if __name__ == '__main__':
    b = Baseline(hidden_size=300, embeds_dim=200)
    print(b)
