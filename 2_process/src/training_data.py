# Read lake-specific sequences from .npz files and save .npz files for training and test data

import os
import numpy as np
import pandas as pd

def split_indices(num_items, split_fractions, seed=0):
    """
    Randomly split a set into non-overlapping subsets by fractions.
    Provide the indices for each new subset.
    Useful for determining a Train/Test or Train/Validate/Test split

    :param num_items: Number of items to split into subsets.
    :param split_fractions: An array-like object containing a fraction for each
        subset. The elements of split_fractions must sum to 1. E.g., for a
        60-20-20 split, split_fractions=[.6, .2, .2]
    :param seed: Random seed used to shuffle indices (Default value = 0)
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

    :param lake_sequence_files: List of lake sequence .npz files
    :param n_depths: Number of discrete depths at which LSTM will output
        temperatures
    :returns: numpy array of sequences

    """
    sequences_list = []
    start_dates_list = []
    site_ids_list = []
    for lake_sequence_file in lake_sequence_files:
        npz = np.load(lake_sequence_file)
        # 1. Lake sequence data
        sequences_list.append(npz['lake_sequences'])
        # 2. Sequence start dates
        lake_start_dates = npz['start_dates']
        # Convert from float32 days to np.datetime64
        lake_start_dates64 = [np.datetime64(int(lsd), 'D') for lsd in lake_start_dates]
        start_dates_list.append(lake_start_dates64)
        # 3. Lake site ID
        lake_sequence_filename = os.path.basename(lake_sequence_file)
        # Format of each filename is "sequences_{site_id}.npz"
        site_id = lake_sequence_filename[10:-4]
        # Shape of sequences is (# sequences, sequence_length, # depths + # input features)
        num_sequences = npz['lake_sequences'].shape[0]
        # Repeat site ID once for every sequence in file
        site_ids_list.append([site_id]*num_sequences)

    sequences = np.concatenate(sequences_list)
    start_dates = np.concatenate(start_dates_list)
    site_ids = np.concatenate(site_ids_list)

    # Remove input features with nans
    # Shape of sequences is (# sequences, sequence_length, # depths + # input features)
    # Slice from n_depths forward to index input features only
    nan_inputs = np.any(np.isnan(sequences[:, :, n_depths:]), axis=(1,2))
    no_nan_inputs = np.invert(nan_inputs)
    return (sequences[no_nan_inputs, :, :],
            start_dates[no_nan_inputs],
            site_ids[no_nan_inputs])


def split_sequence_files(sequences_summary_file,
                         train_frac,
                         valid_frac,
                         test_frac,
                         seed):
    """
    Split lakes into training, validation, and test data.
    
    This function takes lake-specific sequence files and randomly splits them
    into training, validation, and testing sets.

    :param sequences_summary_file: Summary csv with columns sequences_file and num_sequences.
    :param train_frac: The fraction of lakes to put into the training set.
    :param valid_frac: The fraction of lakes to put into the validation set.
    :param test_frac: The fraction of lakes to put into the test set.
    :param seed: Random seed used to shuffle lakes.
    :returns: Tuple of two lists; one list of lake sequence files in the
        training set, and one list of lake sequence files in the test set.

    """
    sequences_summary = pd.read_csv(sequences_summary_file)

    # Remove lakes with 0 sequences
    nonzero_sequences = sequences_summary[sequences_summary.num_sequences>0]
    num_lakes = len(nonzero_sequences)

    # Split lake indices into train, validation, and test
    train_lakes, valid_lakes, test_lakes = split_indices(num_lakes, [train_frac, valid_frac, test_frac], seed=seed)

    train_lake_sequences_files = nonzero_sequences.sequences_file.iloc[train_lakes].to_list()
    valid_lake_sequences_files = nonzero_sequences.sequences_file.iloc[valid_lakes].to_list()
    test_lake_sequences_files = nonzero_sequences.sequences_file.iloc[test_lakes].to_list()
    return (train_lake_sequences_files, valid_lake_sequences_files, test_lake_sequences_files)


def get_train_test_data(train_lake_sequences_files,
                        valid_lake_sequences_files,
                        test_lake_sequences_files,
                        n_depths,
                        n_dynamic,
                        n_static):
    """
    Create training, validation, and test data
    
    This function takes data from lake-specific sequences in .npz files and
    creates arrays for training, validation, and testing.
    1. Read lake-specific sequences from .npz files and concatenate
    2. Standardize using training data
    3. Return data arrays, arrays of sequence start dates, and lake IDs for
       each sequence.

    :param train_lake_sequences_files: List of lake sequence files in the
        training set
    :param valid_lake_sequences_files: List of lake sequence files in the
        validation set
    :param test_lake_sequences_files: List of lake sequence files in the test
        set
    :param n_depths: Number of discrete depths at which LSTM will output
        temperatures
    :param n_dynamic: Number of dynamic input features
    :param n_static: Number of static input features
    :returns: Tuple of five elements
        1. Tuple of 3D Numpy arrays of training data, validation data, and
           testing data
        2. Tuple of Numpy arrays of start dates for sequences in training,
           validation, and testing sets
        3. Tuple of Numpy arrays of lake IDs for sequences in training,
           validation, and testing sets
        4. Numpy array of means of training data (input features and
           temperatures)
        5. Numpy array of standard deviations of training data (input features
           and temperatures)

    """

    # 1. Read lake-specific sequences from .npz files and concatenate

    unscaled_train_data, train_start_dates, train_site_ids = read_sequences(train_lake_sequences_files, n_depths)
    unscaled_valid_data, valid_start_dates, valid_site_ids = read_sequences(valid_lake_sequences_files, n_depths)
    unscaled_test_data, test_start_dates, test_site_ids = read_sequences(test_lake_sequences_files, n_depths)

    # 2. Standardize using training data

    train_data_means = np.nanmean(unscaled_train_data, axis=(0,1))
    train_data_stds = np.nanstd(unscaled_train_data, axis=(0,1))
    # Correct for any 0 standard deviations.
    train_data_stds = np.where(train_data_stds <= 0, 1, train_data_stds)

    # Last (third) dimension of data is of length (# depths + # input features)
    # The means and stds are arrays of length (# depths + # input features)
    # So, numpy will broadcast across the last dimension of unscaled_train_data
    train_data = (unscaled_train_data - train_data_means)/train_data_stds
    valid_data = (unscaled_valid_data - train_data_means)/train_data_stds
    test_data = (unscaled_test_data - train_data_means)/train_data_stds

    # Get site IDs from lake sequence filenames

    # Combine into tuples to reduce number of arguments passed
    split_data = (train_data, valid_data, test_data)
    split_start_dates = (train_start_dates, valid_start_dates, test_start_dates)
    split_site_ids = (train_site_ids, valid_site_ids, test_site_ids)

    return (split_data, 
            split_start_dates,
            split_site_ids,
            train_data_means,
            train_data_stds
           )


def main(sequences_summary_file,
         train_file,
         valid_file,
         test_file,
         split_summary_file,
         process_config):
    """
    Read lake-specific sequences from .npz files, save .npz files for training,
    validation, and test data, and save a file listing which lakes went into
    each set.

    :param sequences_summary_file: Summary csv with columns sequences_file and
        num_sequences.
    :param train_file: training data npz filename with extension
    :param valid_file: validation data npz filename with extension
    :param test_file: test data npz filename with extension
    :param split_summary_file: Filename of csv listing the set for each lake.
    :param process_config: Configuration settings used to form sets. Found in
        2_process/process_config.yaml

    """

    # Split lake sequence files into training, validation, and test sets
    train_lake_sequences_files, valid_lake_sequences_files, test_lake_sequences_files = split_sequence_files(
        sequences_summary_file,
        process_config['train_frac'],
        process_config['valid_frac'],
        process_config['test_frac'],
        process_config['seed']
    )

    # Get training, validation, and test data
    split_data, split_start_dates, split_site_ids, train_data_means, train_data_stds = get_train_test_data(
        train_lake_sequences_files,
        valid_lake_sequences_files,
        test_lake_sequences_files,
        len(process_config['depths_all']),
        len(process_config['dynamic_features_all']),
        len(process_config['static_features_all'])
    )

    # Unpack tuples
    train_data, valid_data, test_data = split_data
    train_start_dates, valid_start_dates, test_start_dates = split_start_dates
    train_site_ids, valid_site_ids, test_site_ids = split_site_ids 

    # Create new directories as needed
    for filepath in [train_file, valid_file, test_file, split_summary_file]:
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Save data to npz along with metadata
    # **process_config saves all configuration settings to npz, with their names as
    # keywords
    np.savez(train_file,
             data=train_data,
             start_dates=train_start_dates,
             site_ids=train_site_ids,
             train_data_means=train_data_means,
             train_data_stds=train_data_stds,
             **process_config)
    np.savez(valid_file,
             data=valid_data,
             start_dates=valid_start_dates,
             site_ids=valid_site_ids,
             train_data_means=train_data_means,
             train_data_stds=train_data_stds,
             **process_config)
    np.savez(test_file,
             data=test_data,
             start_dates=test_start_dates,
             site_ids=test_site_ids,
             train_data_means=train_data_means,
             train_data_stds=train_data_stds,
             **process_config)

    # Save which lakes are in each set: training, validation, or test
    train_lakes_df = pd.DataFrame({'lake_sequences_file':train_lake_sequences_files,
                                   'set':'train'})
    valid_lakes_df = pd.DataFrame({'lake_sequences_file':valid_lake_sequences_files,
                                   'set':'valid'})
    test_lakes_df = pd.DataFrame({'lake_sequences_file':test_lake_sequences_files,
                                   'set':'test'})
    log_df = pd.concat([train_lakes_df, valid_lakes_df, test_lakes_df])
    log_df.to_csv(split_summary_file, index=False, header=False)


if __name__ == '__main__':
    main(snakemake.input['sequences_summary_file'],
         snakemake.output['train_file'],
         snakemake.output['valid_file'],
         snakemake.output['test_file'],
         snakemake.output['split_summary_file'],
         snakemake.params['process_config'])

