import math
from models.slaps.gcn_layers import *
from models.slaps.graph_generators import MLP_GRAPH_GEN

import torch.nn as nn

class SLAPS_GCN_DAE(nn.Module):
    def __init__(self, args, features):
        super().__init__()

        self.layer_count = args.slaps_dae_layers
        self.sparse = args.slaps_sparse_graph

        self.dae_layers = nn.ModuleList()

        self.input_features = args.input_features
        self.hidden_dim = args.slaps_dae_hidden
        self.out_dim = args.slaps_dae_out


        if self.sparse:
            self.dae_layers.append(GCNConv_dgl(self.input_features, self.hidden_dim))
            for _ in range(self.layer_count - 2):
                self.dae_layers.append(GCNConv_dgl(self.hidden_dim, self.hidden_dim))
            self.dae_layers.append(GCNConv_dgl(self.hidden_dim, self.out_dim))
        else:
            self.dae_layers.append(GCNConv_dense(self.input_features, self.hidden_dim))
            for _ in range(self.layer_count - 2):
                self.dae_layers.append(GCNConv_dense(self.hidden_dim, self.hidden_dim))
            self.dae_layers.append(GCNConv_dense(self.hidden_dim, self.out_dim))
        
        self.graph_gen = MLP_GRAPH_GEN(args, features)

        self.dropout = args.slaps_dae_dropout
        self.dropout_adj = nn.Dropout(p=self.dropout)

    
    def forward(self, x):
        adj_ = self.graph_gen(x)    
    
        if self.sparse:
            adj = adj_
            raise NotImplementedError
        else:
            adj = self.dropout_adj(adj_)
        
        for i, convolution in enumerate(self.dae_layers[:-1]):
            x = convolution(x, adj)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.dae_layers[-1](x, adj)

        return x, adj