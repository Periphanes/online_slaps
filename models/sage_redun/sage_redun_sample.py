import torch.nn as nn
import torch

from control.config import args

def aggregate_mean(features, k, l_num):
    """
    Aggregates messages using mean aggregator

    Args:
        features : tensor of feature values
        k : k value in KNN graph generation
        l_num : number of layers
    """

    ret_arr = []

    block_num = pow(k, l_num - 1)
    for i in range(block_num):
        ret_arr.append(torch.mean(features[:, i*k:(i+1)*k, :], 1))
    
    ret_tensor = torch.stack(ret_arr).view((args.batch_size, block_num, -1)).to(args.device)

    return ret_tensor

class SAGE_REDUN_SAMPLE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer_count = args.graph_sample_layers
        self.k = args.knn_k

        self.input_count = args.input_features

        self.intralayer_weights = nn.ModuleList([nn.Linear(self.input_count * 2, self.input_count) for _ in range(self.layer_count)])
        self.relu = nn.ReLU()

        self.lin1 = nn.Linear(self.input_count, 8, bias=True)
        self.bn = nn.BatchNorm1d(8)
        # self.bn = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, args.class_count)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        for i in range(self.layer_count, 0, -1):

            if i != 1:
                concat = torch.concat((aggregate_mean(x[i], self.k, i), x[i-1]), dim=2)
                x[i-1] = self.relu(self.intralayer_weights[i-1](concat))
            else:
                concat = torch.concat((torch.squeeze(aggregate_mean(x[i], self.k, i)), x[i-1]), dim=1)
                x[i-1] = self.relu(self.intralayer_weights[i-1](concat))

        out = self.lin2(self.bn(self.lin1(x[0])))

        return self.sigmoid(out)
