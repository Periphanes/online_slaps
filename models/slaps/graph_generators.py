import torch
import torch.nn as nn

from utils.knn_graph.slaps_utils import *

class MLP_GRAPH_GEN(nn.Module):
    def __init__(self, args, train_data):
        super().__init__()
        self.args = args

        self.mlp_layer_count = args.slaps_mlp_layers
        self.mlp_out = args.slaps_mlp_out
        self.mlp_hidden = args.slaps_mlp_hidden
        self.input_features = args.input_features
        self.mlp_epochs = args.slaps_mlp_epochs

        self.knn_k = args.knn_k
        self.knn_metric = args.slaps_knn_metric
        self.sparse = args.slaps_sparse

        self.features = train_data[0]

        self.mlp_layers = nn.ModuleList()
        if self.mlp_layer_count == 1:
            self.mlp_layers.append(nn.Linear(self.input_features, self.mlp_out))
        else:
            self.mlp_layers.append(nn.Linear(self.input_features, self.mlp_hidden))
            for _ in range(self.mlp_layer_count - 2):
                self.mlp_layers.append(nn.Linear(self.mlp_hidden, self.mlp_hidden))
            self.mlp_layers.append(nn.Linear(self.mlp_hidden, self.mlp_out))
        
        self.knn_k = args.knn_k
        self.mlp_knn_init()

    def mlp_forward(self, x):
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)

            if i != (len(self.mlp_layers) - 1):
                x = nn.functional.relu(x)
        
        return x

    def mlp_knn_init(self):
        if self.input_features == self.mlp_out:
            print("MLP Full")
            for layer in self.mlp_layers:
                layer.weight = nn.Parameter(torch.eye(self.input_features))
        else:
            optimizer = torch.optim.Adam(self.parameters(), 0.01)
            labels = torch.from_numpy(nearest_neighbors(self.features.cpu(), self.knn_k, self.knn_metric)).cuda()

            for epoch in range(1, self.mlp_epochs):
                self.train()
                logits = self.forward(self.features)

                loss = nn.functional.mse_loss(logits, labels, reduction="sum")

                if epoch % 10 == 0:
                    print("MLP Loss", loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        self.features = None
    
    def forward(self, features):
        if self.sparse:
            pass
        else:
            embeddings = self.mlp_forward(features).to(self.args.device)
            embeddings = nn.functional.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.knn_k + 1)
            similarities = nn.functional.relu(similarities)
            return similarities