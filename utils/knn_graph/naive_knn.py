import torch
import numpy as np

from tqdm import tqdm

def naive_knn_gen(args, k, nodes : torch.Tensor, labels : torch.Tensor, train_mask=None, val_mask=None, test_mask=None, test_gen = False):
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

    print("Initializing kNN Graph Construction...")

    if train_mask != None:
        train_nodes = torch.nonzero(train_mask).view(-1)
    else:
        train_nodes = torch.LongTensor([i for i in range(nodes.shape[0])])
    
    edge_list = []

    nodes = nodes.to(args.device)

    for i in tqdm(range(train_nodes.shape[0])):
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

    print("kNN Graph with", train_nodes.shape[0], "nodes and", train_nodes.shape[0] * k, "edges constructed!...")

    return edges


def silent_naive_knn_gen(args, k, nodes : torch.Tensor, labels : torch.Tensor, train_mask=None, val_mask=None, test_mask=None, test_gen = False):
    if train_mask != None:
        train_nodes = torch.nonzero(train_mask).view(-1)
    else:
        train_nodes = torch.LongTensor([i for i in range(nodes.shape[0])])
    
    edge_list = []

    nodes = nodes.to(args.device)

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