from torch.utils.data import Dataset

class feature_Dataset(Dataset):
    def __init__(self, args, X, y, data_type="dataset"):
        self._X_list = X
        self._y_list = y

    def __len__(self):
        return len(self._y_list)
    
    def __getitem__(self, index):
        return self._X_list[index], self._y_list[index]