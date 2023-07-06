import random
import os

from data.collate_fn import *
from data.dataset import *
from torch.utils.data import DataLoader

from data.data_wrangling import *
from utils.knn_graph.naive_knn import naive_knn_gen

import glob_var

def get_data_loader(args):
    print("Initializing Data Loader and Datasets")

    if "facebookpagepage" in args.trainer:
        glob_var.train_data_list, glob_var.val_data_list, glob_var.test_data_list = wrangle_facebookpagepage(args)
    elif "cora" in args.trainer:
        glob_var.train_data_list, glob_var.val_data_list, glob_var.test_data_list = wrangle_cora(args)

    if "sampling" in args.trainer:
        glob_var.train_knn_edges = naive_knn_gen(args, args.knn_k, glob_var.train_data_list[0], glob_var.train_data_list[1], test_gen=True)

    if args.trainer == "basic_train":
        train_data      = basic_Dataset(args, data=glob_var.train_data_list, data_type="training dataset")
        val_data        = basic_Dataset(args, data=glob_var.val_data_list, data_type="validation dataset")
        test_data       = basic_Dataset(args, data=glob_var.test_data_list, data_type="testing dataset")
    if args.trainer == "facebookpagepage_features":
        train_data      = facebook_pagepage_training_Dataset(args, data=glob_var.train_data_list, data_type="training dataset")
        val_data        = facebook_pagepage_training_Dataset(args, data=glob_var.val_data_list, data_type="validation dataset")
        test_data       = facebook_pagepage_training_Dataset(args, data=glob_var.test_data_list, data_type="testing dataset")
    if args.trainer == "facebookpagepage_sampling":
        train_data      = facebook_pagepage_sampling_Dataset(args, data=glob_var.train_data_list, data_type="training dataset")
        val_data        = facebook_pagepage_sampling_test_Dataset(args, data=glob_var.val_data_list, data_type="validation dataset")
        test_data = []
    
    print("Total of {} data points intialized in Training Dataset...".format(train_data.__len__()))
    print("Total of {} data points intialized in Validation Dataset...".format(val_data.__len__()))
    print("Total of {} data points intialized in Testing Dataset...".format(test_data.__len__()))

    if args.trainer == "basic_train":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
    elif args.trainer == "facebookpagepage_features":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
    elif args.trainer == "facebookpagepage_sampling":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_sampling)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_test_sampling)
        test_loader     = None
    
    return train_loader, val_loader, test_loader