import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import sys
sys.path.insert(1, '3_model/src')
from models import Model


class SequenceDataset(Dataset):
    """
    Custom Dataset class for sequences that include dynamic and static inputs,
    and multiple outputs

    __len__ returns the number of sequences
    
    __getitem__ returns dynamic input features, static input features, and
    temperature observations at all depths for one sequence

    Example:

    >>> npz = np.load("2_process/out/mntoha/train.npz")
    >>> n_dynamic = len(npz['dynamic_features_all'])
    >>> n_static = len(npz['static_features_all'])
    >>> n_depths = len(npz['depths'])
    >>> data = torch.from_numpy(npz['data'])
    >>> dataset = SequenceDataset(data, n_depths, n_dynamic, n_static)                            
    >>> # number of sequences
    >>> len(dataset)
    19069
    >>> # dynamic features, static features, and temperatures at depths 
    >>> # for first sequence
    >>> for e in dataset[0]:
    >>>     print(e.shape)
    >>> 
    torch.Size([400, 9])
    torch.Size([400, 4])
    torch.Size([400, 49])

    """
    def __init__(self, sequences, n_outputs, n_dynamic, n_static):
        """
        :param sequences: An array of sequences with shape (# sequences, sequence_length, # depths + # input features)
        :param n_outputs: Number of outputs (# depths)
        :param n_dynamic: Number of dynamic input features
        :param n_static: Number of static input features

        """
        # The number of sequences is the size of the first dimension
        self.len = sequences.shape[0]
        self.n_outputs = n_outputs
        self.n_dynamic = n_dynamic
        self.n_static = n_static
        self.sequences = sequences

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # sequence is organized as temperatures at multiple depths, then dynamic features, then static features
        temperatures = self.sequences[idx, :, :self.n_outputs]
        dynamic_features = self.sequences[idx, :, self.n_outputs:self.n_outputs + self.n_dynamic]
        static_features = self.sequences[idx, :, -self.n_static:]
        return dynamic_features, static_features, temperatures


