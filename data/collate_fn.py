import pickle
import os

import torch
import numpy as np

from control.config import args
from utils.graph_sampling.neighborhood_sample import neighborhood_sample, neighborhood_sample_single
from utils.graph_sampling.neighborhood_project import neighborhood_project_single

import random

import glob_var

def collate_basic(train_data):
    X_batch = []
    y_batch = []

    for data_point in train_data:
        X = data_point[0]
        y = data_point[1]
        X_batch.append(torch.Tensor(X))
        y_batch.append(y)
    
    X = torch.stack(X_batch)
    y = torch.tensor(y_batch)

    return X, y

def collate_fbpp_train_random(train_data):
    '''
    Collate Function for args.trainer == facebookpagepage_features
    Training Time Collate
    '''
    X_batch = []
    y_batch = []

    for data_point in train_data:
        X_batch.append(torch.Tensor(data_point[0]))
        y_batch.append(data_point[1])
    
    X = torch.stack(X_batch)
    y = torch.tensor(y_batch)

    return X, y

def collate_fbpp_train_sampling(train_data):
    X = []
    y = []
    i = 0

    for data_point in train_data:

        # Sample Approximately from the Training Batch
        if random.randint(0, 100) < (100 - args.sage_sample_percent) and i > 1:
            continue

        i += 1

        graph_node, graph_edge = neighborhood_sample_single(args, data_point[0][1], glob_var.train_data_list[0], glob_var.train_knn_edges,
                                                  glob_var.train_data_list[1], masked=False)
        X.append((graph_node, graph_edge))
        y.append(data_point[1])
    
    y = torch.tensor(y)

    return X, y

def collate_fbpp_test_sampling(test_data):
    X = []
    y = []
    
    for data_point in test_data:
        graph_node, graph_edge = neighborhood_project_single(args, data_point[0], glob_var.train_data_list[0], glob_var.train_knn_edges,
                                                             data_point[1], masked=False)
        
        X.append((graph_node, graph_edge))
        y.append(data_point[1])
    
    y = torch.tensor(y)

    return X, y