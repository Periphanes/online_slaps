from torch.utils.data import Dataset
from utils.knn_graph.naive_knn import silent_naive_knn_gen, naive_knn_gen
from utils.graph_sampling.neighborhood_sample import neighborhood_sample_single

from utils.graph_sampling.neighborhood_sample_redun import neighborhood_sample_redun_single

import torch
import random

class feature_Dataset(Dataset):
    def __init__(self, args, X, y, data_type="dataset"):
        self._X_list = torch.tensor(X)
        self._y_list = torch.tensor(y)

    def __len__(self):
        return len(self._y_list)
    
    def __getitem__(self, index):
        return self._X_list[index], self._y_list[index]
    
class relational_staticTrain_Dataset(Dataset):
    def __init__(self, args, X, y, data_type="dataset"):
        self._X_list = torch.tensor(X)
        self._y_list = torch.tensor(y)
        self.args = args

        self._edges = silent_naive_knn_gen(args, args.knn_k, self._X_list, self._y_list, test_gen=True)

        # print(self._edges.shape)
    
    def __len__(self):
        return self.args.relational_train_length
    
    def data_len(self):
        return self._X_list.shape[0]
    
    # def __getitem__(self, index):
    #     node_id = random.randint(0, self.data_len() - 1)
    #     ret_nodes, ret_edges = neighborhood_sample_single(self.args, node_id, self._X_list, self._edges, self._y_list, masked=False)

    #     return ret_nodes, ret_edges, self._y_list[index]

    def __getitem__(self, index):
        node_id = random.randint(0, self.data_len() - 1)
        ret_arr = neighborhood_sample_redun_single(self.args, node_id, self._X_list, self._edges)

        return ret_arr