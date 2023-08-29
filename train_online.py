import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from control.config import args
from models import get_model
from data.online_data.data_preprocess import get_data_loader

import glob_var

from sklearn.metrics import classification_report

glob_var.init()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.cpu or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

print("Device Used : ", device)
args.device = device

if args.dataset == "credit_fraud":
    if args.train_type == "features":
        args.trainer = "credit_features"
    elif args.train_type == "relational":
        args.trainer = "credit_relational"
    else:
        raise NotImplementedError("Trainer type for Credit Fraud not Defined")

elif args.dataset == "ticket_cancel":
    if args.train_type == "features":
        args.trainer = "ticket_features"
    elif args.train_type == "relational":
        args.trainer = "ticket_relational"
    else:
        raise NotImplementedError("Trainer type for Ticket Cancel not Defined")

else:
    raise NotImplementedError("Dataset not defined yet")

train_loader, val_loader, test_loader = get_data_loader(args)

model = get_model(args)
model = model(args).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Parameter Count :", pytorch_total_params)

criterion = nn.BCELoss()