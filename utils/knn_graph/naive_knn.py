import torch
import numpy as np

def naive_knn_gen(args, k, nodes : torch.Tensor, labels : torch.Tensor, train_mask, val_mask, test_mask, test_gen = False):
    """
    Naive kNN Graph Generation Function -> O(n^2) complexity via naive similarity comparisons
    No optimizations applied

    kNN Graph is generated using only training data
    Edge lists for validation and test data generated using training data based kNN graph

    if Test/Validation kNN Generation Wanted, set test_gen to True

    Args:
        args (Namespace): arguments
        k (int): Number of Nearest Neighbors Enumeration
        nodes (torch.Tensor): x feature Data for Graph. Shape of (node_count, feature_count)
        labels (torch.Tensor): y Data for Graph. Shape of (node_count)
        train_mask (torch.Tensor): Mask for Training Data Nodes. Shape of (node_count)
        val_mask (torch.Tensor): Mask for Validation Data Nodes
        test_mask (torch.Tensor): Mask for Testing Data Nodes
    """
    train_nodes = torch.nonzero(train_mask).view(-1)
    edge_list = []

    for i in range(train_nodes.shape[0]):
        cur_node = train_nodes[i].item()
        cur_embd = nodes[cur_node, :].unsqueeze(0)

        sim = torch.nn.CosineSimilarity(dim = 1)
        total_sim = sim(cur_embd, nodes)

        if test_gen:
            top_sim_indices = torch.topk(total_sim, k+1)[1][1:]
        else:
            masked_total_sim = total_sim.masked_fill_(~train_mask, -2)
            top_sim_indices = torch.topk(masked_total_sim, k+1)[1][1:]
            
        edge_list.append(top_sim_indices)
    
    edges = torch.stack(edge_list)

    return edges