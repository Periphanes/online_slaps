import torch
import torch.nn as nn

from models.sage.aggregators import MeanAggregator

class SAGE_CONCAT_LAYER(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.agg = MeanAggregator(args)
        self.weight = nn.Linear(args.input_features * 2, args.input_features)

    def forward(self, x):
        embs = []

        for g in x:
            mp_emb = self.agg(g[0], g[1])
            old_emb = g[0]

            embs.append((mp_emb, old_emb))

        return embs
    
class SAGE_CONCAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_feature_count = args.input_features
        self.args = args
        
        self.sage_layers = SAGE_CONCAT_LAYER(args)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.lin1 = nn.Linear(self.input_feature_count, 64)
        self.lin2 = nn.Linear(self.input_feature_count, 64)

        self.out_lin = nn.Linear(128, args.class_count)
    
    def forward(self, x):
        mp = self.sage_layers(x)
        
        old = []
        new = []

        for i in range(len(x)):
            old.append(x[i][0][0,:])
            new.append(x[i][1][0,:])
        
        old = torch.stack(old).to(self.args.device)
        new = torch.stack(new).to(self.args.device)

        old = self.relu(self.lin1(old))
        new = self.relu(self.lin2(old))

        out = self.out_lin(torch.cat((old, new)))

        return self.softmax(out)