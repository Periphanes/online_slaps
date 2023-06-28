import random
import os

from data.collate_fn import *
from data.dataset import *
from torch.utils.data import DataLoader

from data.data_wrangling import *

def get_data_loader(args):
    print("Initializing Data Loader and Datasets")

    if args.trainer == "facebookpagepage_random":
        train_data_list, val_data_list, test_data_list = wrangle_facebookpagepage(args)

    # train_data_list = []
    # val_data_list = []
    # test_data_list = []

    if args.trainer == "basic_train":
        train_data      = basic_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = basic_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = basic_Dataset(args, data=test_data_list, data_type="testing dataset")
    if args.trainer == "facebookpagepage_random":
        train_data      = facebook_pagepage_training_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = facebook_pagepage_training_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = facebook_pagepage_training_Dataset(args, data=test_data_list, data_type="testing dataset")
    
    print("Total of {} data points intialized in Training Dataset...".format(train_data.__len__()))
    print("Total of {} data points intialized in Validation Dataset...".format(val_data.__len__()))
    print("Total of {} data points intialized in Testing Dataset...".format(test_data.__len__()))

    if args.trainer == "basic_train":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
    if args.trainer == "facebookpagepage_random":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fbpp_train_random)
    
    return train_loader, val_loader, test_loader