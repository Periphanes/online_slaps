import torch
import torch.nn as nn
import numpy as np

import math

class Aggregator(nn.Module):
    def __init__(self, args):
        self.dimension = args.input_features
        self.args = args
    
    # def forward(self, features, nodes, mapping, rows)