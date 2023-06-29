import torch
import torch.nn as nn
import torch.nn.functional as F

class BASELINE_MLP2(nn.Module):
    def __init__(self, args):
        super().__init__()

        input_features = args.input_features
        class_count = args.class_count
        # print(input_features)

        self.hidden1 = nn.Linear(input_features, 256)
        self.hidden2 = nn.Linear(256, 512)
        self.hidden3 = nn.Linear(512, 64)
        self.output = nn.Linear(64, class_count)
        self.softmax = nn.Softmax()

    def forward(self, x):
        em1 = self.hidden1(x)
        em2 = self.hidden2(em1)
        em3 = self.hidden3(em2)
        out = self.output(em3)

        return self.softmax(out)
    