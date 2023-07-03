import torch
import numpy as np

from utils.graph_sampling.neighborhood_sample import neighborhood_sample_single

def neighborhood_project_single(args, query : torch.Tensor, nodes, edges, labels, masked=True, train_mask=None,
                                val_mask=None, test_mask=None, flow_type="val", edge_type="adj"):
    graph_sample_layers = args.graph_sample_layers
    knn_k = args.knn_k

    sim = torch.nn.CosineSimilarity(dim = 1)
    total_sim = sim(query, nodes)
    
    query_edges = torch.topk(total_sim, knn_k)[1].to(args.device)
    node_index = nodes.shape[0]

    node_cat = torch.cat((nodes, query.unsqueeze(dim=0)))
    edges_cat = torch.cat((edges, query_edges.unsqueeze(dim=0)))

    graph_nodes, graph_edges = neighborhood_sample_single(args, node_index, node_cat, edges_cat, labels, masked=masked,
                                                          train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, flow_type=flow_type,
                                                          edge_type=edge_type)
    
    return graph_nodes, graph_edges