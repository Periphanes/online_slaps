from torch.utils.data import DataLoader

import pickle
import numpy as np

from data.online_data.dataset import *
from data.online_data.collate_fn import *

from sklearn.preprocessing import StandardScaler

def get_data_loader(args):
    print("Initializing Data Loader and Datasets")

    if args.dataset == "ticket_cancel":
        pickle_dir = 'datasets/ticket_cancel/ticket_preprocessed.pickle'
        with open(pickle_dir, 'rb') as handle:
            X, y = pickle.load(handle)
        
        args.input_features = 17
        args.class_count = 1
        
    elif args.dataset == "credit_fraud":
        pickle_dir = 'datasets/credit_fraud/credit_preprocessed.pickle'
        with open(pickle_dir, 'rb') as handle:
            X, y = pickle.load(handle)
        
        args.input_features = 29
        args.class_count = 1
    
    data_len = y.shape[0]

    train_len = int(data_len * 0.6)
    test_len = int(data_len * 0.8)

    train_X = X[:train_len]
    val_X = X[train_len:test_len]
    test_X = X[test_len:]

    train_y = y[:train_len]
    val_y = y[train_len:test_len]
    test_y = y[test_len:]

    train_X = np.nan_to_num(train_X)
    val_X = np.nan_to_num(val_X)
    test_X = np.nan_to_num(test_X)

    std = StandardScaler()
    std.fit(train_X)

    train_X = std.transform(train_X)
    val_X = std.transform(val_X)
    test_X = std.transform(test_X)

    if args.trainer == "ticket_features":
        train_data      = feature_Dataset(args, train_X, train_y, data_type="train dataset")
        val_data        = feature_Dataset(args, val_X, val_y, data_type="validation dataset")
        test_data       = feature_Dataset(args, test_X, test_y, data_type="test dataset")
    
    if args.trainer == "ticket_relational":
        train_data      = relational_staticTrain_Dataset(args, train_X, train_y, data_type="train dataset")
        val_data        = relational_staticTrain_Dataset(args, val_X, val_y, data_type="validation dataset")
        test_data       = relational_staticTrain_Dataset(args, test_X, test_y, data_type="test dataset")
    
    # print("Total of {} data points intialized in Train Dataset...".format(train_data.__len__()))
    # print("Total of {} data points intialized in Validation Dataset...".format(val_data.__len__()))
    # print("Total of {} data points intialized in Test Dataset...".format(test_data.__len__()))

    if args.trainer == "ticket_features":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        val_loader      = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        test_loader     = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    if args.trainer == "ticket_relational":
        train_loader    = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_relational_redun_train)
        val_loader      = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader     = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return train_loader, val_loader, test_loader