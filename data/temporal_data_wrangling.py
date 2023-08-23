import torch
import numpy as np

from torch_geometric.datasets import JODIEDataset

def wrangle_jodie():
    jodie = JODIEDataset(root='./datasets/jodie', name='wikipedia').to_datapipe()
    jodie = jodie.batch_graphs(batch_size=1)

    for batch in jodie:
        graph = batch[0]
    
    print(graph)

    print(graph['src'][:100])

wrangle_jodie()