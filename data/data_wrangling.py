import torch
import numpy as np

import torch_geometric

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import FacebookPagePage

# from control.config import args

def wrangle_facebookpagepage(args):
    fb = FacebookPagePage(root='./datasets/facebook').to_datapipe()
    fb = fb.batch_graphs(batch_size =1)

    for batch in fb:
        graph = batch[0]
    
    nodes = graph["x"]
    labels = torch.unsqueeze(graph["y"], 1)

    concat = torch.cat((nodes, labels), 1)
    # print(concat.shape)

    random_indexes = torch.randperm(concat.shape[0])
    shuffled_data = concat[random_indexes]

    data_len = shuffled_data.shape[0]

    X_data = shuffled_data[:, :128]
    y_data = shuffled_data[:, 128]

    # print(X_data.shape)
    # print(y_data.shape)

    train_limit = int(data_len * 0.6)
    val_limit = int(data_len * 0.8)
    y_data = y_data.squeeze()

    train_data_list = (X_data[:train_limit, :], y_data[:train_limit])
    val_data_list = (X_data[train_limit:val_limit, :], y_data[train_limit:val_limit])
    test_data_list = (X_data[val_limit:, :], y_data[val_limit:])

    args.input_features = 128
    args.class_count = 4

    return train_data_list, val_data_list, test_data_list