import torch
from torch import nn
import torch.nn.functional as F

class FF_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.linear(x))

    def goodness(self, x):
        return (x ** 2).mean(dim=1)

class FF_Network(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            FF_Layer(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])

    def forward_pass(self, x):
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
