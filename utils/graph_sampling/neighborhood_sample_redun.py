import torch
import numpy as np

def neighborhood_sample_redun_single(args, node_index, nodes, edges):
    graph_sample_layers = args.graph_sample_layers

    ret_arr = [nodes[node_index]]

    outliers = [node_index]

    for layer_num in range(1, graph_sample_layers + 1):
        new_outliers = []
        feature_agg = []

        for outlier in outliers:
            edge_list = edges[outlier]

            for edge in edge_list:
                edge_ind = edge.item()
                feature_agg.append(nodes[edge_ind])
                new_outliers.append(edge_ind)
        
        outliers = new_outliers
        ret_arr.append(torch.stack(feature_agg))

    return ret_arr