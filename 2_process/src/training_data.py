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
    # Convert split_fractions to 1D numpy array
    split_fractions = np.array(split_fractions).flatten()
    num_splits = len(split_fractions)

    # Make sure the fractions sum to 1
    fraction_sum = split_fractions.sum()
    # Allow for floating point precision error by using np.isclose
    if not np.isclose(fraction_sum, 1.0):
        raise ValueError("split_fractions must sum to 1")

    # Create an array of indices from 0 to num_items - 1
    item_indices = np.arange(num_items)
    # Shuffle its order
    rng = np.random.default_rng(seed)
    rng.shuffle(item_indices)

    # Determine where to split the array item_indices so that each split has
    # the correct fraction of the total items.
    # Running sum of split fractions
    cumulative_fractions = split_fractions.cumsum()
    # Multiply the running sum by the total number of items and round to an
    # integer to get the ending index of each split
    where_to_split = (num_items*cumulative_fractions).round().astype(int)
    # Insert 0 at the beginning so that we can use the array for slicing
    where_to_split = np.insert(where_to_split, 0, 0)
    # The first split will be where_to_split[0]:where_to_split[1]
    # The second split will be where_to_split[1]:where_to_split[2]
    # And so on.

    # Split item_indices, using where_to_split to slice
    split_indices = [item_indices[where_to_split[i_split]:where_to_split[i_split+1]] 
                     for i_split in range(num_splits)]
    return split_indices


def read_sequences(lake_sequence_files, n_depths):
    """
    Read a list of sequence files in, remove sequences that have any NaN values
    among their input features, and return a single numpy array containing all
    sequences from all sequence files.

    :param lake_sequence_files: List of lake sequence .npy files
    :param n_depths: Number of discrete depths at which LSTM will
    output temperatures
    :returns: numpy array of sequences

    """
    sequences = np.concatenate([np.load(sfn) for sfn in lake_sequence_files])
    # remove input features with nans
    # shape of sequences is (# sequences, sequence_length, # depths + # input features)
    # slice from n_depths forward to index input features only
    nan_inputs = np.any(np.isnan(sequences[:, :, n_depths:]), axis=(1,2))
    return sequences[np.invert(nan_inputs), :, :]


def get_train_test_sequences_files(sequences_summary_file,
                                   train_frac,
                                   test_frac,
                                   seed):
    """
    Split lakes into train and test data.

    This function takes lake-specific sequence files and randomly splits them
    into training and testing sets.

    :param sequences_summary_file: Summary csv with columns sequences_file and num_sequences.
    :param train_frac: The fraction of lakes to put into the training set.
    :param test_frac: The fraction of lakes to put into the test set.
    :param seed: Random seed used to shuffle lakes.
    :returns: Tuple of two lists; one list of lake sequence files in the
        training set, and one list of lake sequence files in the test set.

    """
    sequences_summary = pd.read_csv(sequences_summary_file)

    # Remove lakes with 0 sequences
    nonzero_sequences = sequences_summary[sequences_summary.num_sequences>0]
    num_lakes = len(nonzero_sequences)

    # Split lake indices into train and test
    train_lakes, test_lakes = split_indices(num_lakes, [train_frac, test_frac], seed=seed)

    train_lake_sequences_files = nonzero_sequences.sequences_file.iloc[train_lakes].to_list()
    test_lake_sequences_files = nonzero_sequences.sequences_file.iloc[test_lakes].to_list()
    return (train_lake_sequences_files, test_lake_sequences_files)


def get_train_test_data(train_lake_sequences_files,
                        test_lake_sequences_files,
                        n_depths,
                        n_dynamic,
                        n_static):
    """
    Create train and test data

    This function takes data from lake-specific sequences in .npy files and creates arrays for training and testing.

    1. Read lake-specific sequences from .npy files and concatenate
    2. Standardize using training data

    """

    # 1. Read lake-specific sequences from .npy files and concatenate

    unscaled_train_data = read_sequences(train_lake_sequences_files, n_depths)
    unscaled_test_data = read_sequences(test_lake_sequences_files, n_depths)

    # 2. Standardize using training data

    train_data_means = np.nanmean(unscaled_train_data, axis=(0,1))
    train_data_stds = np.nanstd(unscaled_train_data, axis=(0,1))
    # Correct for any 0 standard deviations.
    train_data_stds = np.where(train_data_stds <= 0, 1, train_data_stds)

    # Last (third) dimension of data is of length (# depths + # input features)
    # The means and stds are arrays of length (# depths + # input features)
    # So, numpy will broadcast across the last dimension of unscaled_train_data
    train_data = (unscaled_train_data - train_data_means)/train_data_stds
    test_data = (unscaled_test_data - train_data_means)/train_data_stds

    return (train_data, test_data, train_data_means, train_data_stds)


if __name__ == '__main__':

    # Split lake sequence files into train and test sets
    train_lake_sequences_files, test_lake_sequences_files = get_train_test_sequences_files(
        snakemake.input['sequences_summary_file'],
        snakemake.params['train_frac'],
        snakemake.params['test_frac'],
        snakemake.params['seed']
    )

    # Get training and test data
    train_data, test_data, train_data_means, train_data_stds = get_train_test_data(
        train_lake_sequences_files,
        test_lake_sequences_files,
        snakemake.params['n_depths'],
        snakemake.params['n_dynamic'],
        snakemake.params['n_static']
    )

    # Output files to save
    npz_file_train = snakemake.output['train_file']
    npz_file_test = snakemake.output['test_file']
    train_test_summary_file = snakemake.output['train_test_summary_file']

    # Save data to npz along with metadata
    # **snakemake.params saves all snakemake params to npz, using their names
    # as keywords
    np.savez(npz_file_train,
             data=train_data,
             train_data_means=train_data_means,
             train_data_stds=train_data_stds,
             **snakemake.params)
    np.savez(npz_file_test,
             data=test_data,
             train_data_means=train_data_means,
             train_data_stds=train_data_stds,
             **snakemake.params)

    # Save which lakes are in the training set and which are in the test set
    train_lakes_df = pd.DataFrame({'lake_sequences_file':train_lake_sequences_files,
                                   'set':'train'})
    test_lakes_df = pd.DataFrame({'lake_sequences_file':test_lake_sequences_files,
                                   'set':'test'})
    log_df = pd.concat([train_lakes_df, test_lakes_df])
    log_df.to_csv(train_test_summary_file, index=False, header=False)

