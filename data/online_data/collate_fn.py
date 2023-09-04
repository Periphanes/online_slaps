import torch
from control.config import args

def collate_relational_train(train_data):
    node_batch = []
    edge_batch = []
    y_batch = []

    # print(len(train_data))

    for data_point in train_data:
        node_batch.append(data_point[0])

        int_edge = [x.type(torch.LongTensor) for x in data_point[1]]

        edge_batch.append(int_edge)
        y_batch.append(data_point[2])

    y_batch = torch.stack(y_batch)

    return node_batch, edge_batch, y_batch

def collate_relational_redun_train(train_data):
    k = args.knn_k
    layers = args.graph_sample_layers

    y_batch = []

    batched = [[] for _ in range(layers + 1)]

    for data_point, y in train_data:
        for ind, data in enumerate(data_point):
            batched[ind].append(data)
        
        y_batch.append(y)

    y_batch = torch.stack(y_batch)
    batched = [torch.stack(x) for x in batched]

    return batched, y_batch