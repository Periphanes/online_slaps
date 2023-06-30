import os
import random

import torch

import pickle
from tqdm import tqdm

class basic_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []

        self._data_list = data
    
    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        return self._data_list[index]

class facebook_pagepage_training_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = data

    def __len__(self):
        return self._data_list[1].shape[0]
    
    def __getitem__(self, index):
        return (self._data_list[0][index, :], self._data_list[1][index])

class facebook_pagepage_sampling_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = data
    
    def __len__(self):
        return self._data_list[1].shape[0]
    
    def __getitem__(self, index):
        return ((self._data_list[0][index, :], index), self._data_list[1][index])
    
class facebook_pagepage_sampling_test_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = data
    
    def __len__(self):
        return self._data_list[1].shape[0]
    
    def __getitem__(self, index):
        return (self._data_list[0][index, :], self._data_list[1][index])
