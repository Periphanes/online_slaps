import torch
import torch.nn as nn
import numpy as np

import math

class Aggregator(nn.Module):
    def __init__(self, args):
        self.dimension = args.input_features
        self.args = args
    
    # def forward(self, features, nodes, mapping, rows)

class MeanAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, nodes, edges):
        agg_feat = []

        node_count = nodes.shape[0]

        for i in range(node_count):
            mean_feat = torch.zeros_like(nodes[0, :]).to(self.args.device)
            for j in edges[i]:
                torch.add(mean_feat, nodes[j, :])
            mean_feat = torch.div(mean_feat, node_count)
            agg_feat.append(mean_feat)
        
        agg_feat = torch.stack(agg_feat)

        return agg_feat