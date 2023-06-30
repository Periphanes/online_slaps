import torch
import torch.nn as nn

class SAGE_TEST(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_feature_count = args.input_features
        self.hid1 = nn.Linear(self.input_feature_count, 23)

    def forward(self, x):
        print(x[1])
        em1 = self.hid1(x)
        exit(0)