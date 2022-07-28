# Train an LSTM or EA-LSTM and save the results with metadata

import os
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam

import sys
sys.path.insert(1, '3_train/src')
from models import Model


class SequenceDataset(Dataset):
    """
    Custom Torch Dataset class for sequences that include dynamic and static inputs
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


def get_data_loaders(train_ds, valid_ds, batch_size):
    """
    Get Torch DataLoaders for both the train Dataset and the validate Dataset
    
    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param train_ds: Torch Dataset for training data
    :param valid_ds: Torch Dataset for validation data
    :param batch_size: Number of elements in each training batch
    :returns: Tuple of training data loader and validation data loader

    """
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        # The validate set is not used for training with backprop, so we can
        # afford its batch size to be larger and it doesn't need to be shuffled
        DataLoader(valid_ds, batch_size=batch_size * 2),
    )


def get_sequence_dataset(npz_filepath,
                         dynamic_features_use,
                         static_features_use,
                         depths_use):
    """
    Get SequenceDataset from npz file
    
    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param npz_filepath: Name and path to .npz data file
    :param dynamic_features_use: List of dynamic features to include
    :param static_features_use: List of static features to include
    :param depths_use: List of depth values to include

    """
    # Load data
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

    dataset = SequenceDataset(
        data_array_to_use,
        n_depths,
        n_dynamic,
        n_static
    )
    return dataset


def get_data(train_npz_filepath,
             valid_npz_filepath,
             dynamic_features_use,
             static_features_use,
             depths_use,
             batch_size):
    """
    Get train and validate dataloaders from npz file
    
    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param train_npz_filepath: Name and path to training data .npz file
    :param valid_npz_filepath: Name and path to validation data .npz file
    :param dynamic_features_use: List of dynamic features to include
    :param static_features_use: List of static features to include
    :param depths_use: List of depth values to include
    :param batch_size: Number of elements in each training batch

    """

    # Get SequenceDataset (custom Dataset) objects for training and validation datasets
    train_subset = get_sequence_dataset(
        train_npz_filepath,
        dynamic_features_use,
        static_features_use,
        depths_use)
    valid_subset = get_sequence_dataset(
        valid_npz_filepath,
        dynamic_features_use,
        static_features_use,
        depths_use)
    return get_data_loaders(train_subset, valid_subset, batch_size)


def get_model(n_depths,
              n_dynamic,
              n_static,
              hidden_size,
              initial_forget_bias,
              dropout,
              concat_static):
    """
    Create LSTM or EA-LSTM torch model
    
    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-optim

    :param n_depths: Number of depths at which to predict temperatures (number
        of outputs at each time step)
    :param n_dynamic: Number of dynamic input features
    :param n_static: Number of static input features
    :param hidden_size: Number of elements in hidden state and cell state
    :param initial_forget_bias: Value of the initial forget gate bias
    :param dropout: Dropout probability, from 0 to 1 (0 = don't use dropout)
    :param concat_static: If True, uses standard LSTM. Otherwise, uses EA-LSTM
    :returns: LSTM or EA-LSTM torch model

    """
    model = Model(
        input_size_dyn=n_dynamic,
        input_size_stat=n_static,
        hidden_size=hidden_size,
        output_size=n_depths,
        initial_forget_bias=initial_forget_bias,
        dropout=dropout,
        concat_static=concat_static
    )

    return model


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
    # There should be at least one finite value in y
    if not loss_idx.any():
        raise ValueError('All temperature labels in this batch are not finite'
                         ' (NaN, infinity, or negative infinity)')
    loss = loss_func(pred[loss_idx], y[loss_idx])

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    # NaN in the loss can cause training to fail silently
    # Better to fail loudly!
    if not torch.isfinite(loss):
        raise ValueError(f'Batch loss is not finite. Its value is {loss}.')

    # return loss, number of finite values in y
    return loss.item(), torch.sum(loss_idx).item()


def fit(max_epochs,
        model,
        loss_func,
        opt,
        train_dl,
        valid_dl,
        device,
        weights_filepath,
        early_stopping_patience,
        verbose=False):
    """
    Train the model, and compute training and validation losses for each epoch
    
    Patterned after https://pytorch.org/tutorials/beginner/nn_tutorial.html#create-fit-and-get-data

    :param max_epochs: Maximum number of training epochs
    :param model: PyTorch model inheriting nn.Module()
    :param loss_func: PyTorch loss function, e.g., nn.MSELoss()
    :param opt: Optimizer to use to update model parameters
    :param train_dl: PyTorch DataLoader with training data
    :param valid_dl: PyTorch DataLoader with validation data
    :param device: Device that pytorch is running on (cpu or gpu)
    :param weights_filepath: Path and filename to save model weights to
    :param early_stopping_patience: Number of epochs to train without
        validation loss improvement, i.e., training stops after
        early_stopping_patience epochs without improvement in the validation
        loss. Set early_stopping_patience to -1 to turn off early stopping.
    :param verbose: Print more messages during execution
        (Default value = False)
    :returns: Tuple of:
        array of training losses for each epoch,
        array of validation losses for each epoch,
        last epoch at which weights were saved

    """

    # Count number of non-NaN observations in training and validation sets
    n_train = 0
    for x_d, x_s, y in train_dl:
        n_train += torch.sum(torch.isfinite(y))
    n_valid = 0
    for x_d, x_s, y in valid_dl:
        n_valid += torch.sum(torch.isfinite(y))

    def verbose_print(*args):
        # Only print if verbose is true
        if verbose:
            print(*args, flush=True)

    train_losses = []
    valid_losses = []
    print('Epoch: train loss, validate loss', flush=True)
    num_epochs_without_improvement = 0
    saved_epoch = -1
    # Training loop
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        # Data is ordered as [dynamic inputs, static inputs, labels]
        verbose_print('Train loss by batch')
        for x_d, x_s, y in train_dl:
            # Transfer to device
            x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
            batch_loss, batch_count = loss_batch(model, loss_func, x_d, x_s, y, opt)

            # Weight the batch loss based on number of observations in each batch
            verbose_print(batch_loss, batch_count)
            train_loss += batch_loss * batch_count/n_train

        # Loss for each validation batch, with number of obs per batch
        model.eval()
        valid_loss = 0.0
        verbose_print('Validation loss by batch')
        with torch.no_grad():
            for x_d, x_s, y in valid_dl:
                # Transfer to device
                x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
                batch_loss, batch_count = loss_batch(model, loss_func, x_d, x_s, y)

                # Weight the batch loss based on number of observations in each batch
                verbose_print(batch_loss, batch_count)
                valid_loss += batch_loss * batch_count/n_valid

        print(f'{epoch}: {datetime.now()} {train_loss}, {valid_loss}', flush=True)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if early_stopping_patience != -1:
            # If this model has the lowest validation loss, save model weights
            if valid_loss == min(valid_losses):
                num_epochs_without_improvement = 0
                saved_epoch = epoch
                save_weights(model, weights_filepath, overwrite=True)
            # Otherwise, add to count of high validation loss models and check for early stopping condition
            else:
                num_epochs_without_improvement += 1
                # An early_stopping_patience of -1 means early stopping should not happen
                if num_epochs_without_improvement >= early_stopping_patience:
                    print(f'Stopping at epoch {epoch} after {num_epochs_without_improvement} epochs without improvement', flush=True)
                    break

    if early_stopping_patience == -1:
        saved_epoch = epoch
        save_weights(model, weights_filepath, overwrite=True)
    return train_losses, valid_losses, saved_epoch


def save_weights(model, filepath, overwrite=True):
    """
    Save weights of a torch model
    
    Optionally, append a unique number to the end of the filename to avoid
    overwriting any existing files.

    :param model: PyTorch model to be saved
    :param filepath: Path and filename to save to
    :param overwrite: If True, overwrite existing file if necessary. If False,
        append a unique suffix to the filename before saving.
        (Default value = True)

    """
    # Create new directory if needed
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not overwrite:
        # Append unique suffix to filename
        root, extension = os.path.splitext(filepath)
        suffix = 0
        while os.path.exists(filepath):
            suffix += 1
            filepath = f'{root}_{suffix}{extension}'
    torch.save(model.state_dict(), filepath)


def save_metadata(config, train_npz_filepath, valid_npz_filepath, save_filepath, overwrite=True):
    """
    Save configuration settings and metadata
    
    Combine the metadata in config with the metadata in the npz
    file and save the combination to a new output npz file.
    
    Optionally, append a unique number to the end of the filename to avoid
    overwriting any existing files.

    :param config: Dictionary of configuration settings and training results to save
    :param train_npz_filepath: Name and path to .npz data file containing training data
    :param valid_npz_filepath: Name and path to .npz data file containing validation data
    :param save_filepath: Path and filename to save to
    :param overwrite: If True, overwrite existing file if necessary. If False,
        append a unique suffix to the filename before saving.
        (Default value = True)

    """
    # Create new directory if needed
    directory = os.path.dirname(save_filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not overwrite:
        # Append unique suffix to filename
        root, extension = os.path.splitext(save_filepath)
        suffix = 0
        while os.path.exists(save_filepath):
            suffix += 1
            save_filepath = f'{root}_{suffix}{extension}'

    # Save all metadata in npzs
    train_npz = np.load(train_npz_filepath)
    valid_npz = np.load(valid_npz_filepath)
    # ...but not the training data itself
    train_npz.files.remove('data')
    # Combine training data settings and config into a new metadata dictionary
    metadata = {}
    # The start dates and the site IDs are different for training and validation data
    # Save start_dates and site_ids for both train and valid
    metadata['train_start_dates'] = train_npz['start_dates']
    metadata['valid_start_dates'] = valid_npz['start_dates']
    metadata['train_site_ids'] = train_npz['site_ids']
    metadata['valid_site_ids'] = valid_npz['site_ids']
    # Remove start_dates and site_ids so that they don't get added to metadata twice
    train_npz.files.remove('start_dates')
    train_npz.files.remove('site_ids')
    # The rest of the metadata in train_npz (training data means and standard 
    # deviations, process_config parameters) is identical to that in valid_npz
    # So, we can add it from train_npz alone without any loss of info
    for key in train_npz:
        metadata[key] = train_npz[key]
    # Any duplicate keys get overwritten by the value in `config` (the train_config)
    for key in config:
        metadata[key] = config[key]
    np.savez(save_filepath, **metadata)


def get_git_hash():
    """
    Get the hash of the current git revision
    
    From https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

    :returns: String of current git revision's hash

    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_status():
    """
    Get the output of the `git status` command

    :returns: String of output of `git status`

    """
    # status = str(subprocess.Popen(['git status'], cwd = None if code_dir=="" else code_dir,shell=True,stdout=subprocess.PIPE).communicate()[0]).split("\\n")
    return subprocess.check_output(['git', 'status']).decode('ascii').strip()


def main(train_npz_filepath, 
         valid_npz_filepath,
         weights_filepath,
         metadata_filepath,
         run_id,
         model_id,
         config):
    """
    Train a model and save the trained weights
    
    Load training and validation data from .npz files. Use the settings
    specified in the config dictionary. Train the model, and save its weights.

    :param train_npz_filepath: Name and path to training data .npz file
    :param valid_npz_filepath: Name and path to validation data .npz file
    :param weights_filepath: Path and filename to save weights to
    :param metadata_filepath: Path and filename to save metadata to
    :param run_id: ID of the current training run (experiment). Value must
        match the run_id as defined in the config dictionary.
    :param model_id: ID of the model to train within the current training run
    :param config: Dictionary of configuration settings, including:
        run_id                string, ID of current run (experiment)
        run_description       string, plain language description of current run
        max_epochs            integer, Maximum number of epochs to train for
        loss_criterion        string, Name of class in torch.nn to use for loss
        learning_rate         float, Learning rate of optimizer
        concat_static         boolean, Add static to dynamic and use standard LSTM, or not
        dropout               float, Dropout probability, from 0 to 1 (0 = don't use dropout)
        initial_forget_bias   float, Value of the initial forget gate bias
        hidden_size           integer, Number of elements in hidden state and cell state
        static_features_use   list of strings, Names of static features for model to use
        dynamic_features_use  list of strings, Names of dynamic features for model to use
        depths_use            list of floats, depth values of temperatures for model to use
        batch_size            integer, Number of examples per training batch

    """
    # Check that run_id in config matches output path
    if not (run_id == config['run_id']):
        raise ValueError(f"The values of 'run_id' do not match. 'run_id' in process_config.yaml is {config['run_id']}, but 'run_id' in the output filepath is {run_id}.")

    # Set seed to encourage reprodicibility
    torch.manual_seed(config['seed'])

    # Get objects for training
    # model, loss function, optimizer, training data loader, validation data loader

    # Create dataloaders
    train_data_loader, valid_data_loader = get_data(
        train_npz_filepath,
        valid_npz_filepath,
        config['dynamic_features_use'],
        config['static_features_use'],
        config['depths_use'],
        config['batch_size']
    )

    n_depths = len(config['depths_use'])
    n_dynamic = len(config['dynamic_features_use'])
    n_static = len(config['static_features_use'])

    if torch.cuda.is_available():
        device = "cuda" 
    else:
        device = "cpu"
    print(f"Using {device} device", flush=True)

    print(f"Number of threads set by user: {torch.get_num_threads()}", flush=True)
    print(f"Number of GPUs: {torch.cuda.device_count()}", flush=True)
    print(f"Number of CPUs: {os.cpu_count()}", flush=True)

    # Create model
    model = get_model(n_depths,
                      n_dynamic,
                      n_static,
                      config['hidden_size'],
                      config['initial_forget_bias'],
                      config['dropout'],
                      config['concat_static']).to(device)

    # Create optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Get loss function
    # Equivalent to loss_func = nn.MSELoss()
    criterion_class = getattr(nn, config['loss_criterion'])
    loss_func = criterion_class()

    # Training loop
    train_start_time = str(datetime.now())
    train_losses, valid_losses, saved_epoch = fit(
        config['max_epochs'],
        model,
        loss_func,
        optimizer,
        train_data_loader,
        valid_data_loader,
        device,
        weights_filepath,
        config['early_stopping_patience']
    )
    train_end_time = str(datetime.now())
    print('Finished Training', flush=True)

    # Save model settings and training metrics
    config['model_id'] = model_id
    config['train_start_time'] = train_start_time
    config['train_end_time'] = train_end_time
    config['train_losses'] = train_losses
    config['valid_losses'] = valid_losses
    config['n_epochs_trained'] = len(train_losses)
    config['saved_epoch'] = saved_epoch
    config['git_hash'] = get_git_hash()
    config['git_status'] = get_git_status()
    save_metadata(config, train_npz_filepath, valid_npz_filepath, metadata_filepath, overwrite=True)


if __name__ == '__main__':
    main(snakemake.input.train_npz_filepath,
         snakemake.input.valid_npz_filepath,
         snakemake.output.weights_filepath,
         snakemake.output.metadata_filepath,
         snakemake.params.run_id,
         snakemake.params.model_id,
         snakemake.params.config)

