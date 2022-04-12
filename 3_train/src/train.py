import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

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
    
    :Example:


    >>> npz = np.load("2_process/out/mntoha/train.npz")
    >>> n_depths = len(npz['depths'])
    >>> n_dynamic = len(npz['dynamic_features_all'])
    >>> n_static = len(npz['static_features_all'])
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
        self.len = len(sequences)
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


def split(dataset, fraction, seed):
    """
    Randomly split a dataset into two non-overlapping subsets

    :param dataset: A torch.utils.data.Dataset to be split
    :param fraction: The fraction of data samples to put in the first subset
    :param seed: Seed for random number generator
    :returns: Tuple of two datasets for the two subsets

    """
    # split into two subsets
    n_total = len(dataset)
    n_1 = int(round(n_total * fraction))
    n_2 = n_total - n_1
    subset_1, subset_2 = random_split(
        dataset, [n_1, n_2], generator=torch.Generator().manual_seed(seed))
    return subset_1, subset_2


def get_data_loaders(train_ds, valid_ds, batch_size):
    """
    Get data loaders for both the train dataset and the validate dataset

    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param train_ds: Dataset for training data
    :param valid_ds: Dataset for validation data
    :param batch_size: Number of elements in each training batch
    :returns: Tuple of training data loader and validation data loader

    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        # The validate set is not used for training with backprop, so we can
        # afford its batch size to be larger and it doesn't need to be shuffled
        DataLoader(valid_ds, batch_size=batch_size * 2),
    )


def get_data(npz_filepath,
             dynamic_features_use,
             static_features_use,
             depths_use,
             batch_size,
             valid_frac,
             seed):
    """
    Get train and validate dataloaders from npz file

    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param npz_filepath: Name and path to .npz data file
    :param dynamic_features_use: List of dynamic features to include
    :param static_features_use: List of static features to include
    :param depths_use: List of depth values to include
    :param batch_size: Number of elements in each training batch
    :param valid_frac: The fraction of data samples to put into the validation set
    :param seed: Seed for random number generator

    """

    # Check that valid_frac is a fraction
    if (valid_frac < 0) or (valid_frac > 1):
        raise ValueError("valid_frac must be between 0 and 1")

    # Training data
    data_npz = np.load(npz_filepath)

    # Determine which depths and features to use
    # Input features are drivers, clarity, ice flags, and static attributes, in that order.
    # Combine full set into one list
    elements_all = (data_npz['depths_all'].tolist() +
                    data_npz['dynamic_features_all'].tolist() +
                    data_npz['static_features_all'].tolist())
    # Combine set to use into one list
    elements_use = (depths_use + dynamic_features_use + static_features_use)
    # Select set to use out of full set
    idx_to_use = [elements_all.index(e) for e in elements_use]
    # Shape of data is (# sequences, sequence_length, # depths + # input features)
    data_array_to_use = data_npz['data'][:,:,idx_to_use]

    n_depths = len(depths_use)
    n_dynamic = len(dynamic_features_use)
    n_static = len(static_features_use)

    # Train/validate split
    dataset = SequenceDataset(
        data_array_to_use,
        n_depths,
        n_dynamic,
        n_static
    )
    train_subset, valid_subset = split(dataset, 1-valid_frac, seed)
    return get_data_loaders(train_subset, valid_subset, batch_size)

