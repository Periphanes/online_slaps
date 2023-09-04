import torch
import numpy as np

import copy

def neighborhood_sample(args, node_indexes, nodes, edges, labels, masked=True, train_mask=None, 
                        val_mask=None, test_mask=None, flow_type="train", edge_type="adj"):
    graph_sample_layers = args.graph_sample_layers

    node_batch = []
    edge_batch = []

    for node_index in node_indexes:
        index_count = 1
        node_index = node_index.item()
        index_dict = {node_index : 0}

        graph_nodes = [nodes[node_index]]
        graph_indexes = [node_index]
        graph_edges = []

        outliers = [node_index]
        
        for graph_layer in range(graph_sample_layers):
            new_outliers = []
            
            for outlier in outliers:
                new_nodes = []
                new_indexes = []
                for edge in edges[outlier]:
                    if edge.item() not in index_dict:
                        index_dict[edge.item()] = index_count
                        index_count += 1
                        new_nodes.append(nodes[edge.item()])
                        new_indexes.append(edge.item())
                        new_outliers.append(edge.item())
                graph_nodes += new_nodes
                graph_indexes += new_indexes
            
            outliers = new_outliers
        
        for node in graph_indexes:
            if node in outliers:
                continue
            edge_list = copy.deepcopy(edges[node])
            edge_list.apply_(lambda x: index_dict[x])
            graph_edges.append(edge_list)

        graph_nodes = torch.stack(graph_nodes)
        graph_edges = torch.stack(graph_edges)
    
        node_batch.append(graph_nodes)
        edge_batch.append(graph_edges)
    
    return node_batch, edge_batch

def neighborhood_sample_single(args, node_index, nodes, edges, labels, masked=True, train_mask=None, 
                        val_mask=None, test_mask=None, flow_type="train", edge_type="adj"):
    graph_sample_layers = args.graph_sample_layers

    index_count = 1
    index_dict = {node_index : 0}

    graph_nodes = [nodes[node_index]]
    graph_indexes = [node_index]
    graph_edges = []

    outliers = [node_index]
    
    for _ in range(graph_sample_layers):
        new_outliers = []
        
        for outlier in outliers:
            new_nodes = []
            new_indexes = []
            for edge in edges[outlier]:
                if edge.item() not in index_dict:
                    index_dict[edge.item()] = index_count
                    index_count += 1
                    new_nodes.append(nodes[edge.item()])
                    new_indexes.append(edge.item())
                    new_outliers.append(edge.item())
            graph_nodes += new_nodes
            graph_indexes += new_indexes
        
        outliers = new_outliers
    
    for node in graph_indexes:
        edge_list = edges[node].to('cpu')
        # edge_list.apply_(lambda x: index_dict[x])

        new_edge_list = []

        for edge in edge_list:
            if edge.item() in index_dict:
                new_edge_list.append(index_dict[edge.item()])

        graph_edges.append(torch.tensor(new_edge_list).to(args.device))

    graph_nodes = torch.stack(graph_nodes)

    # print(graph_nodes.shape)
    # print(graph_edges.shape)

    # exit(1)

    # print(graph_nodes.shape)#, graph_edges.shape)
    # exit(0)

    return graph_nodes, graph_edges