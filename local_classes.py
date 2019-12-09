import torch
from torch.utils import data

class Dataset(data.Dataset):
  #Characterizes a dataset for PyTorch
  def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        #Denotes the total number of samples
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Load data and get label
        X = torch.Tensor(self.list_IDs[index])
        # need to read in or do something so this is in a usable type ya feel
        # y = torch.Tensor(self.labels[index])
        y = self.labels[index]
        return X, y
