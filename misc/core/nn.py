import torch
import torch.nn as nn
import torch.nn.functional as F

class NPNet(nn.Module):
    def __init__(self, theta0):
        super().__init__()
        self.para = nn.Parameter(theta0)

    def forward(self, x):
        return self.para
    
