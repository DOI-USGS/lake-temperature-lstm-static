import os
import pandas as pd

def to_csv(in_file, csv_file, file_format):
    """
    Convert file to csv format
    """
    destination_dir = os.path.dirname(csv_file)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if file_format == 'feather':
        pd.read_feather(in_file).to_csv(csv_file)
    else:
        raise ValueError(f'File format {file_format} not recognized')


if __name__ == '__main__':
    to_csv(snakemake.input['in_file'], snakemake.output['csv_file'], snakemake.params['file_format'])
