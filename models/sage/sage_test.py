import torch
import torch.nn as nn

from models.sage.aggregators import MeanAggregator

class SAGE_LAYER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.agg = MeanAggregator(args)
        self.weight = nn.Linear(args.input_features * 2, args.input_features)
        
    def forward(self, x):
        embs = []

        for g in x:
            mp_emb = self.agg(g[0], g[1])
            old_emb = g[0]

            concat_emb = torch.cat((mp_emb, old_emb), dim=1)

            # print(mp_emb.shape, old_emb.shape)
            # print(concat_emb.shape)
            
            embs.append((self.weight(concat_emb), g[1])) 

        return embs

class SAGE_TEST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_feature_count = args.input_features
        self.graph_layer_count = args.graph_sample_layers
        
        self.sage_layers = nn.ModuleList()
        for _ in range(self.graph_layer_count):
            self.sage_layers.append(SAGE_LAYER(args))
        
        self.args = args
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.lin1 = nn.Linear(self.input_feature_count, 64, bias=True)
        self.bn = nn.BatchNorm1d(64)
        self.lin2 = nn.Linear(64, args.class_count)

    def forward(self, x):
        for i, sl in enumerate(self.sage_layers):
            x = sl(x)
        
        out = []

        for i in range(len(x)):
            out.append(x[i][0][0,:])
        
        out = torch.stack(out).to(self.args.device)

        out = self.lin2(self.relu(self.bn(self.lin1(out))))

        return self.softmax(out)