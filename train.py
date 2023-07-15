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
from data.data_wrangling import *
from utils.knn_graph.naive_knn import naive_knn_gen
from utils.graph_sampling.neighborhood_sample import neighborhood_sample

import glob_var

from sklearn.metrics import classification_report


from models.slaps.graph_generators import MLP_GRAPH_GEN

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

# nodes, labels, trm, vam, tem = wrangle_reddit(args) 
# edges = naive_knn_gen(args, 10, nodes, labels, trm, vam, tem)

# tdl, vdl, tedl = wrangle_facebookpagepage(args)
# nodes, labels = tdl

# edges = naive_knn_gen(args, 10, nodes, labels, test_gen=True)

# neighborhood_sample(args, torch.randint(1,20000, (2,)), nodes, edges, labels)

# exit(1)

if args.dataset == "facebookpagepage":
    if args.input_type == "features":
        args.trainer = "facebookpagepage_features"
    elif args.input_type == "sampling":
        args.trainer = "facebookpagepage_sampling"
elif args.dataset == "cora":
    if args.input_type == "features":
        args.trainer = "cora_features"
    elif args.input_type == "sampling":
        args.trainer = "cora_sampling"
    elif args.input_type == "slaps":
        args.trainer = "cora_slaps"
elif args.dataset == "emaileucore":
    if args.input_type == "features":
        args.trainer = "email_features"
    elif args.input_type == "sampling":
        args.trainer = "email_sampling"
else:
    raise NotImplementedError("Trainer Not Defined Yet")

train_loader, val_loader, test_loader = get_data_loader(args)

# test_mlp = MLP_GRAPH_GEN(args, glob_var.train_data_list)

model = get_model(args)
model = model(args).to(device)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Parameter Count :", pytorch_total_params)

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

    input_1 = ["facebookpagepage_features", "cora_features"]
    input_2 = ["facebookpagepage_sampling", "cora_sampling"]
    input_3 = []

    for train_batch in tqdm(train_loader):
        if any(args.trainer in x for x in input_1):
            train_x, train_y = train_batch
            try:
                train_x = train_x.to(device)
            except:
                train_x = [x.to(device) for x in train_x]
        elif any(args.trainer in x for x in input_2):
            train_x, train_y = train_batch
            try:
                train_x = (train_x[0].to(device), train_x[1].to(device))
            except:
                try:
                    train_x = [(x[0].to(device), x[1].to(device)) for x in train_x]
                except:
                    train_x = [(x[0].to(device), x[1]) for x in train_x]
        elif any(args.trainer in x for x in input_3):
            train_x, train_y = train_batch
            try:
                train_x = (train_x[0].to(device), train_x[1].to(device), train_x[2].to(device), train_x[3].to(device))
            except:
                train_x = [(x[0].to(device), x[1].to(device), x[2].to(device)) for x in train_x]
        
        try:
            train_y = train_y.to(device)
        except:
            train_y = [y.to(device) for y in train_y]

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

        input_1 = ["facebookpagepage_features", "cora_features"]
        input_2 = ["facebookpagepage_sampling", "cora_sampling"]
        input_3 = []

        with torch.no_grad():
            for val_batch in val_loader:
                if any(args.trainer in x for x in input_1):
                    val_x, val_y = val_batch
                    try:
                        val_x = val_x.to(device)
                    except:
                        val_x = [x.to(device) for x in val_x]
                elif any(args.trainer in x for x in input_2):
                    val_x, val_y = val_batch
                    try:
                        val_x = (val_x[0].to(device), val_x[1].to(device))
                    except:
                        try:
                            val_x = [(x[0].to(device), x[1].to(device)) for x in val_x]
                        except:
                            val_x = [(x[0].to(device), x[1]) for x in val_x]
                elif any(args.trainer in x for x in input_3):
                    val_x, val_y = val_batch
                    try:
                        val_x = (val_x[0].to(device), val_x[1].to(device), val_x[2].to(device), val_x[3].to(device))
                    except:
                        val_x = [(x[0].to(device), x[1].to(device), x[2].to(device)) for x in val_x]
                
                try:
                    val_y = val_y.to(device)
                except:
                    val_y = [y.to(device) for y in val_y]

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