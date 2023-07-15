import torch
import numpy as np

import torch_geometric

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import FacebookPagePage
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import CoraFull
from torch_geometric.datasets import EmailEUCore

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

def wrangle_reddit(args):
    fb = Reddit(root='./datasets/reddit').to_datapipe()
    fb = fb.batch_graphs(batch_size = 1)

    for batch in fb:
        graph = batch[0]
    
    nodes = graph["x"]
    labels = torch.unsqueeze(graph["y"], 1)

    train_mask = graph["train_mask"]
    val_mask = graph["val_mask"]
    test_mask = graph["test_mask"]

    node_feature_count = nodes.shape[1]

    # print(torch.sum(nodes[0]))

    # train_y = torch.masked_select(labels, train_mask)
    # val_y = torch.masked_select(labels, val_mask)
    # test_y = torch.masked_select(labels, test_mask)

    # enlarged_train_mask = train_mask.expand(-1, node_feature_count)

    return nodes, labels, train_mask, val_mask, test_mask

def wrangle_cora(args):
    cr = CoraFull(root='./datasets/cora').to_datapipe()
    cr = cr.batch_graphs(batch_size=1)

    for batch in cr:
        graph = batch[0]

    nodes = graph["x"]
    labels = torch.unsqueeze(graph["y"], 1)

    concat = torch.cat((nodes, labels), 1)

    random_indexes = torch.randperm(concat.shape[0])
    shuffled_data = concat[random_indexes]

    data_len = shuffled_data.shape[0]

    X_data = shuffled_data[:, :128]
    y_data = shuffled_data[:, 128]
    
    train_limit = int(data_len * 0.6)
    val_limit = int(data_len * 0.8)
    y_data = y_data.squeeze()

    train_data_list = (X_data[:train_limit, :], y_data[:train_limit])
    val_data_list = (X_data[train_limit:val_limit, :], y_data[train_limit:val_limit])
    test_data_list = (X_data[val_limit:, :], y_data[val_limit:])

    args.input_features = 128
    args.class_count = 4

    return train_data_list, val_data_list, test_data_list

# No node features
def wrangle_emaileucore(args):
    em = EmailEUCore(root = './datasets/email').to_datapipe()
    em = em.batch_graphs(batch_size = 1)

    for batch in em:
        print(batch[0])
        exit(0)
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