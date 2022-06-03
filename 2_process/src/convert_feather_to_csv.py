import os
import pandas as pd

def feather_to_csv(in_file, csv_file):
    """
    Convert a feather file to csv format.

    :param in_file: Path to feather file to convert
    :param csv_file: Path to csv to save

    """
    destination_dir = os.path.dirname(csv_file)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    pd.read_feather(in_file).to_csv(csv_file, index=False)


if __name__ == '__main__':
    feather_to_csv(snakemake.input['in_file'], snakemake.output['csv_file'])
