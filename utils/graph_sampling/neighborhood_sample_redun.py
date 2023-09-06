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
        feat_agg_tensor = torch.stack(feature_agg)

        # print("\nNAN/INF")
        # print(torch.count_nonzero(torch.isnan(feat_agg_tensor)))
        # print(torch.count_nonzero(torch.isinf(feat_agg_tensor)))

        feat_agg_tensor = torch.nan_to_num(feat_agg_tensor)

        ret_arr.append(feat_agg_tensor)

    return ret_arr