import os
import zipfile
import numpy as np
import pandas as pd

def unzip_files(files, source_dir, destination_dir):
    """
    Unzip files and save them to another directory. Files are saved 

    :param files: Files to unzip, with full paths
    :param source_dir: Directory to pull zipped files from. 
        This part of the path is removed from the beginning of files to determine where to save unzipped files.
    :param destination_dir: Directory to save unzipped files to.
        In each file path, source_dir is replaced with destination_dir to determine where to save unzipped files.
    :returns: list of paths to unzipped files

    """
    destination_files = []
    for f in files:
        # check that files are inside source_dir
        if os.path.samefile(source_dir, os.path.commonpath([f, source_dir])):
            relfile = os.path.relpath(f, source_dir)
            destination_file = os.path.join(destination_dir, os.path.splitext(relfile)[0])
            with zipfile.ZipFile(f, 'r') as zf:
                zf.extractall(destination_file)
            destination_files.append(destination_file)
        else:
            raise FileNotFoundError(f'File {f} not in directory {source_dir}')
    return destination_files


def assemble_data(lake_metadata):
    """
    TODO: Docstring for assemble_data.

    :param lake_metadata: TODO
    :returns: TODO

    """
    pass

def main(in_files, out_file, config):
    """
    Sort all files

    :param out_file: file to save processed data to
    :returns: ScienceBase JSON response

    """

    filename = os.path.basename(out_file)
    destination_dir = os.path.dirname(out_file)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # unzip files to 2_process/tmp folder
    zip_files = [f for f in in_files if f.lower().endswith('.zip')]
    unzipped_files = unzip_files(zipfiles, source_dir, destination_dir)

    obs = pd.read_csv(config['obs_file'])
    lake_metadata = pd.read_csv(config['metadata_file'])

    


    # response = sb_get(sb, item_id, filename, destination_dir=destination_dir)
    np.savez(out_file, np.array([1,2,3]))


print(snakemake.input[0], snakemake.output[0])
if __name__ == '__main__':
    main(snakemake.input, snakemake.output[0], snakemake.params['config'])

