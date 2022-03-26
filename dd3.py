import torch
from torch import nn
from torch.nn import functional as F


class ddNet(nn.Module):

    def __init__(self):
        super(ddNet, self).__init__()

        # xw+b
        self.fc0 = nn.Linear(9, 200, bias=False)
        self.dd = nn.Linear(200, 200, bias=False)
        self.dd2 = nn.Linear(200, 200, bias=False)
        self.dd3 = nn.Linear(200, 200, bias=False)
        self.dd4 = nn.Linear(200, 200, bias=False)

        self.fc2 = nn.Linear(200, 1, bias=False)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        x = self.fc0(x)
        c = x
        # h1 = x@w1*x
        
        x=self.dd(x)*c
        x=self.dd2(x)*c
        x=self.dd3(x)*c
        x=self.dd4(x) * c

        x = self.fc2(x)
        return x
