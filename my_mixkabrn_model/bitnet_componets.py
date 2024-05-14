import torch
import torch.nn as nn

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        binarized_weight = torch.sign(self.weight)  # Binarize the weights
        return nn.functional.linear(x, binarized_weight) * self.scale

