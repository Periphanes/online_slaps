import torch

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