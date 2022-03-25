# Read lake-specific sequences from .npy files and save .npz files for training and test data

import numpy as np
import pandas as pd

def split_indices(num_items, split_fractions, seed=0):
    """
    Randomly split a set into non-overlapping subsets by fractions.
    Provide the indices for each new subset.
    Useful for determining a Train/Test or Train/Validate/Test split

    :param num_items: Number of items to split into subsets.
    :param split_fractions: An array-like object containing a fraction for each subset. The elements of split_fractions must sum to 1.
        E.g., for a 60-20-20 split, split_fractions=[.6, .2, .2]
    :param seed: Random seed used to shuffle indices
    :returns: A list of arrays of indices, one for each subset.

    """


def read_sequences(lake_sequence_files, n_depths):
    """
    Read a list of sequence files in, remove sequences that have any
    NaN values among their inputs, and return a single numpy array
    containing all sequences from all sequence files.

    :param lake_sequence_files: List of lake sequence .npy files
    :param n_depths: Number of discrete depths at which LSTM will
    output temperatures
    :returns: numpy array of sequences

    """


def scaling_params(data, **kwargs):
    """
    Get means and standard deviations of an array of data. Ignore NaN
    values. Replace any standard deviations of 0 with 1.

    :param data: Array-like set of data
    :param **kwargs: Keyword arguments used by np.nanmean and np.nanstd

    """


def get_train_test_data(sequences_summary_file,
                        train_frac,
                        test_frac,
                        n_depths,
                        n_dynamic,
                        n_static,
                        seed):
    """
    Create train and test data

    This function takes data from lake-specific sequences in .npy files and creates arrays for training and testing.

    1. Split lakes into train and test sets
    2. Read lake-specific sequences from .npy files and concatenate
    3. Normalize using training data

    """

    # 1. Split lakes into train and test sets
    sequences_summary = pd.read_csv(sequences_summary_file)

    # Remove lakes with 0 sequences
    nonzero_sequences = sequences_summary[sequences_summary.num_sequences>0]
    num_lakes = len(nonzero_sequences)

    # Split lake indices into train and test
    train_lakes, test_lakes = split_indices(num_lakes, [train_frac, test_frac], seed=seed)

    lake_sequences_files_train = nonzero_sequences.sequences_file.iloc[train_lakes]
    lake_sequences_files_test = nonzero_sequences.sequences_file.iloc[test_lakes]

    # 2. Read lake-specific sequences from .npy files and concatenate

    unscaled_train_data = read_sequences(lake_sequences_files_train, n_depths)
    unscaled_test_data = read_sequences(lake_sequences_files_test, n_depths)

    # 3. Normalize using training data

    feature_means, feature_stds = scaling_params(
        unscaled_train_data,
        axis=(0,1)
    )

    # Last dimension of data is features
    train_data = (unscaled_train_data - feature_means)/feature_stds
    test_data = (unscaled_test_data - feature_means)/feature_stds

    return (train_data, test_data, feature_means, feature_stds)

