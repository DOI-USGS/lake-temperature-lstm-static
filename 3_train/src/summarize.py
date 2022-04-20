import os
import glob
import numpy as np
import pandas as pd

# Don't write these out to the summary file (they're long lists)
files_to_exclude = [
    'train_data_means',
    'train_data_stds',
    'train_losses',
    'valid_losses'
]

def read_npz_as_dataframe(filename):
    npz = np.load(filename)
    # Make every item in the dict a one-element list so that pandas can read it in easily
    metadata_dict = {file: [npz[file]] for file in npz.files if not file in files_to_exclude}
    df = pd.DataFrame.from_dict(metadata_dict, orient='columns')
    return df


def main(summary_filepath):

    metadata_file_pattern = '*.npz'
    # Search the directory that will contain the summary file 
    # for all metadata npz files, recursively
    summary_dir = os.path.dirname(summary_filepath)
    glob_str = summary_dir + os.sep + '**' + os.sep + metadata_file_pattern
    metadata_files = glob.glob(glob_str, recursive=True)

    # Read all metadata npzs into a list of DataFrames
    dfs = [read_npz_as_dataframe(fn) for fn in metadata_files]
    # Concatenate list of DataFrames into one DataFrame
    summary_df = pd.concat(dfs, sort=False)
    # run_id should be a unique identifier
    summary_df.set_index('run_id', inplace=True)
    summary_df.to_csv(summary_filepath)

if __name__ == '__main__':
    main(snakemake.output.summary_filepath)

