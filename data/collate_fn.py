import pickle
import os

import torch
import numpy as np

from control.config import args

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