from models.slaps.gcn_layers import *
from models.slaps.graph_generators import MLP_GRAPH_GEN

import torch.nn as nn

class SLAPS_GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList()

        self.sparse = args.slaps_sparse_graph
        self.layer_count = self.slaps_gcn_layers

        self.input_features = args.input_features
        self.hidden_dim = args.slaps_gcn_hidden
        self.out_dim = args.slaps_gcn_out

        if self.sparse:
            self.gcn_layers.append(GCNConv_dgl(self.input_features, self.hidden_dim))
            for _ in range(self.layer_count - 2):
                self.gcn_layers.append(GCNConv_dgl(self.hidden_dim, self.hidden_dim))
            self.gcn_layers.append(GCNConv_dgl(self.hidden_dim, self.out_dim))
        else:
            self.gcn_layers.append(GCNConv_dense(self.input_features, self.hidden_dim))
            for _ in range(self.layer_count - 2):
                self.gcn_layers.append(GCNConv_dense(self.hidden_dim, self.hidden_dim))
            self.gcn_layers.append(GCNConv_dense(self.hidden_dim, self.out_dim))

        self.dropout = args.slaps_dae_dropout
        self.dropout_adj = nn.Dropout(p=self.dropout)
    
    def forward(self, x, adj):
        pass