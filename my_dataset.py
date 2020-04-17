import torch
from torch.utils import data

class Dataset(data.Dataset):

  def __init__(self, corpus, labels):
        'Initialization'
        self.labels = labels
        self.corpus = corpus

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.corpus)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        return self.corpus[index], self.labels[index]
