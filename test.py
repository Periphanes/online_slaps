import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch_geometric

from torch_geometric.datasets import JODIEDataset
from torch_geometric.datasets import FacebookPagePage

# dp = JODIEDataset(root='./data/jodie', name='reddit').to_datapipe()
# dp = dp.batch_graphs(batch_size=2, drop_last = True)

# for batch in dp:
#     print(batch)
#     exit(1)

# dp = JODIEDataset(root='./data/jodie', name='reddit')

# print(dp[0])



fb = FacebookPagePage(root='./data/facebook').to_datapipe()
fb = fb.batch_graphs(batch_size =1)

for batch in fb:
    graph = batch[0]


print(graph["y"])