from torch.utils.data import DataLoader

import pickle
import glob_var

def get_data_loader(args):
    print("Initializing Data Loader and Datasets")

    if args.dataset == "ticket_cancel":
        pickle_dir = '../../datasets/ticket_cancel/ticket_preprocessed.pickle'
        with open(pickle_dir, 'rb') as handle:
            X, y = pickle.load(handle)
        
    elif args.dataset == "credit fraud":
        pickle_dir = '../../datasets/credit_fraud/credit_preprocessed.pickle'
        with open(pickle_dir, 'rb') as handle:
            X, y = pickle.load(handle)
    
    data_len = y.shape[0]
    static_len = int(data_len * 0.8)

    static_X = X[:static_len]
    online_X = X[static_len:]

    static_y = y[:static_len]
    online_y = y[static_len:]

    