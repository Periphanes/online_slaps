import random
import os

from data.collate_fn import *
from data.dataset import *
from torch.utils.data import DataLoader

def get_data_loader(args):
    print("Initializing Data Loader and Datasets")

    train_data_list = []
    val_data_list = []
    test_data_list = []

    if args.trainer == "basic_train":
        train_data      = basic_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data        = basic_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data       = basic_Dataset(args, data=test_data_list, data_type="testing dataset")
    
    print("Total of {} data points intialized in Training Dataset...".format(train_data.__len__()))
    print("Total of {} data points intialized in Validation Dataset...".format(val_data.__len__()))
    print("Total of {} data points intialized in Testing Dataset...".format(test_data.__len__()))

    if args.trainer == "basic_train":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_basic)
    
    return train_loader, val_loader, test_loader