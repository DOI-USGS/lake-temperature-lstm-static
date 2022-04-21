import os
import glob
import numpy as np
import pandas as pd

# Don't write these out to the summary file (they're long lists)
FILES_TO_EXCLUDE = [
    'train_data_means',
    'train_data_stds',
    'train_losses',
    'valid_losses'
]

def read_npz_as_dataframe(npz_name):
    """
    Read a numpy npz and convert to a pandas DataFrame.

    Exclude specific files that are inside the npz from the DataFrame because
    they are long lists that don't need to be saved to a summary file.

    :param npz_name: Name of npz to read
    :returns: Pandas DataFrame containing contents of npz, with the names of
        the files in the npz as column names

    """
    npz = np.load(npz_name)
    # Make every item in the dict a one-element list so that pandas can read it in easily
    metadata_dict = {file: [npz[file]] for file in npz.files if not file in FILES_TO_EXCLUDE}
    df = pd.DataFrame.from_dict(metadata_dict, orient='columns')
    return df


def main(summary_filepath):
    """
    Summarize all trained models in all subdirectories of a given root directory.

    Read in all .npz files in the subdirectories of the root and assume that
    all .npz files are metadata files saved to accompany trained models.
    Extract information to summarize from those .npz files, and save it to a
    csv. Use the combination of the run_id and the model_id to create a unique
    identifier for every trained model, and save that in the first column of
    the summary csv. 

    :param summary_filepath: File path of the summarizing csv to create

    """
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
    # Use combination of run_id and model_id as a unique index
    summary_df['full_id'] = summary_df['run_id'].astype(str) + '.' + summary_df['model_id'].astype(str)
    summary_df.set_index('full_id', inplace=True)
    summary_df.to_csv(summary_filepath)

if __name__ == '__main__':
    main(snakemake.output.summary_filepath)

