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


def get_model(
    n_depths,
    n_dynamic,
    n_static,
    hidden_size,
    initial_forget_bias,
    dropout,
    concat_static,
    learning_rate
):
    """
    Return LSTM model and optimizer

    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-optim

    :param n_depths: Number of depths at which to predict temperatures (number
        of outputs at each time step)
    :param n_dynamic: Number of dynamic input features
    :param n_static: Number of static input features
    :param hidden_size: Number of elements in hidden state and cell state
    :param initial_forget_bias: Value of the initial forget gate bias
    :param dropout: Dropout probability, from 0 to 1
    :param concat_static: If True, uses standard LSTM. Otherwise, uses EA-LSTM
    :param learning_rate: Learning rate for optimizer
    :returns: Tuple of model and optimizer

    """
    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu"
    print(f"Using {device} device")

    model = Model(
        input_size_dyn=n_dynamic,
        input_size_stat=n_static,
        hidden_size=hidden_size,
        output_size=n_depths,
        initial_forget_bias=initial_forget_bias,
        dropout=dropout,
        concat_static=concat_static
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def loss_batch(model, loss_func, x_d, x_s, y, opt=None):
    """
    Evaluate loss for one batch of inputs and outputs, and optionally update model
    parameters by backpropagation

    Non-finite (e.g. NaN) values in y are ignored.

    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param model: PyTorch model inheriting nn.Module()
    :param loss_func: PyTorch loss function, e.g., nn.MSELoss()
    :param x_d: Dynamic input features
    :param x_s: Static input features
    :param y: Outputs (temperatures at multiple depths)
    :param opt: Optimizer to use to update model parameters. If opt=None, do
        not update model parameters (Default value = None)
    :returns: Tuple of loss value and number of finite (non-NaN) output labels
        in batch

    """

    # model only needs one set of static inputs
    # They are identical at all time steps
    pred, h, c = model(x_d, x_s[:, 0, :])
    # Only compute loss where y is finite
    loss_idx = torch.isfinite(y)
    loss = loss_func(pred[loss_idx], y[loss_idx])

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    # return loss.item(), len(y)
    return loss.item(), torch.sum(loss_idx)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    Train the model, and compute training and validation losses for each epoch

    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param epochs: Maximum number of training epochs
    :param model: PyTorch model inheriting nn.Module()
    :param loss_func: PyTorch loss function, e.g., nn.MSELoss()
    :param opt: Optimizer to use to update model parameters
    :param train_dl: PyTorch DataLoader with training data
    :param valid_dl: PyTorch DataLoader with validation data

    """

    # Count number of observations in training and validation sets
    n_train = 0
    for x_d, x_s, y in train_dl:
        n_train += torch.sum(torch.isfinite(y))
    n_valid = 0
    for x_d, x_s, y in valid_dl:
        n_valid += torch.sum(torch.isfinite(y))

    train_losses = []
    valid_losses = []
    print('Epoch: train loss, validate loss')
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        # data is ordered as [dynamic inputs, static inputs, labels]
        for x_d, x_s, y in train_dl:
            batch_loss, batch_count = loss_batch(model, loss_func, x_d, x_s, y, opt)

            # Track this epoch's loss
            train_loss += batch_loss * batch_count/n_train

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, x_d, x_s, y) for x_d, x_s, y in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(f'{epoch}: {train_loss}, {val_loss}')
        train_losses.append(train_loss)
        valid_losses.append(val_loss)

