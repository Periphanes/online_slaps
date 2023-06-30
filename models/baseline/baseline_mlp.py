import torch
import torch.nn as nn
import torch.nn.functional as F

class BASELINE_MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        input_features = args.input_features
        class_count = args.class_count
        # print(input_features)

        self.hidden1 = nn.Linear(input_features, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, class_count)
        self.softmax = nn.Softmax()

    def forward(self, x):
        em1 = self.hidden1(x)
        em2 = self.hidden2(em1)
        out = self.output(em2)

        return self.softmax(out)