import torch
import numpy as np

def neighborhood_project_single(args, query, nodes, edges, labels, masked=True, train_mask=None,
                                val_mask=None, test_mask=None, flow_type="val", edge_type="adj"):
    graph_sample_layers = args.graph_sample_layers
    knn_k = args.knn_k

    sim = torch.nn.CosineSimilarity(dim = 1)
    total_sim = sim(query, nodes)
    
    query_edges = torch.topk(total_sim, knn_k)[1]

    print(query_edges.get_device())
    exit(1)