# Main Training File
import os
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from control.config import args
from models import get_model
from trainer import get_trainer
from data.data_preprocess import get_data_loader

from sklearn.metrics import classification_report

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

if args.dataset == "facebookpagepage":
    args.trainer = "facebookpagepage_random"
else:
    raise NotImplementedError("Trainer Not Defined Yet")

train_loader, val_loader, test_loader = get_data_loader(args)

model = get_model(args)
model = model(args).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=args.lr_init)

iter_num_per_epoch = len(train_loader)
iter_num_total = args.epochs * iter_num_per_epoch

print("# of Iterations (per epoch): ",  iter_num_per_epoch)
print("# of Iterations (total): ",      iter_num_total)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

iteration = 0

pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")

best_validation_loss = 10.0
validation_loss_lst = []

for epoch in range(1, args.epochs+1):

    # Training Step Start
    model.train()

    training_loss = []

    for train_batch in tqdm(train_loader):
        if args.trainer == "facebookpagepage_random":
            train_x, train_y = train_batch
            train_x = train_x.to(device)
            train_y = train_y.to(device)
        elif args.trainer == "two inputs":
            train_x, train_y = train_batch
            train_x = (train_x[0].to(device), train_x[1].to(device))
            train_y = train_y.to(device)            
        elif args.trainer == "three inputs":
            train_x, train_y = train_batch
            train_x = (train_x[0].to(device), train_x[1].to(device), train_x[2].to(device), train_x[3].to(device))
            train_y = train_y.to(device)

        iteration += 1

        model, iter_loss, _, _ = get_trainer(args = args,
                                       iteration = iteration,
                                       x = train_x,
                                       static = None,
                                       y = train_y,
                                       model = model,
                                       device = device,
                                       scheduler=scheduler,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       flow_type="train")

        training_loss.append(iter_loss)

    if epoch % args.val_epoch == 0:

        # Validation Step Start
        model.eval()

        validation_loss = []
        pred_batches = []
        true_batches = []

        with torch.no_grad():
            for val_batch in val_loader:
                if args.trainer == "facebookpagepage_random":
                    val_x, val_y = val_batch
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                elif args.trainer == "two inputs":
                    val_x, val_y = val_batch
                    val_x = (val_x[0].to(device), val_x[1].to(device))
                    val_y = val_y.to(device)
                elif args.trainer == "three inputs":
                    val_x, val_y = val_batch
                    val_x = (val_x[0].to(device), val_x[1].to(device), val_x[2].to(device), val_x[3].to(device))
                    val_y = val_y.to(device)

                model, val_loss, pred, true = get_trainer(args = args,
                                            iteration = iteration,
                                            x = val_x,
                                            static = None,
                                            y = val_y,
                                            model = model,
                                            device = device,
                                            scheduler=scheduler,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            flow_type="val")

                validation_loss.append(val_loss)
                pred_batches.append(pred)
                true_batches.append(true)

            avg_validation_loss = sum(validation_loss) / len(validation_loss)
            validation_loss_lst.append(avg_validation_loss)

            pred = torch.argmax(torch.cat(pred_batches), dim=1).cpu()
            true = torch.cat(true_batches).cpu()
            
            print(classification_report(true, pred))

            pbar.set_description("Training Loss : " + str(sum(training_loss) / len(training_loss)) + " / Val Loss : " + str(avg_validation_loss))
            
    pbar.update(1)
    pbar.refresh()

    # if best_validation_loss > avg_validation_loss:
    #     torch.save(model, './saved_models/best_model.pt')
    #     torch.save(model.audio_feature_extractor, './saved_models/audio_feature_extractor' + str(epoch) + '.pt')
    #     best_validation_loss = avg_validation_loss