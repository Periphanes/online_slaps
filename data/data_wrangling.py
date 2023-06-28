import torch
import numpy as np

import torch_geometric

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import FacebookPagePage

# from control.config import args

def wrangle_facebookpagepage():
    fb = FacebookPagePage(root='./datasets/facebook').to_datapipe()
    fb = fb.batch_graphs(batch_size =1)

    for batch in fb:
        graph = batch[0]
    
    nodes = graph["x"]
    labels = torch.unsqueeze(graph["y"], 1)

    concat = torch.cat((nodes, labels), 1)
    print(concat.shape)

    random_indexes = torch.randperm(concat.shape[0])
    shuffled_data = concat[random_indexes]


wrangle_facebookpagepage()