import torch.nn as nn


class Classifier(nn.Module):
    """
    a single layer Classifier
    """
    def __init__(self, n_hid):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.linear = nn.Linear(n_hid, 1)

    def forward(self, x):
        output = self.linear(x).squeeze()
        output = nn.functional.sigmoid(output) * 9
        return output
