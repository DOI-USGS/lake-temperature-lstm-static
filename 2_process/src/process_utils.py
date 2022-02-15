import os
import zipfile
import requests
import urllib

import numpy as np
import pandas as pd
from scipy import interpolate

def elevation_function(lat, lon):
    """
    Query service using lat, lon and return the corresponding elevation values.
    Thanks to https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python

    :param lat: latitude values in decimal degrees
    :param lon: longitude values
    :returns: elevation at the specified coordinates

    """
    # USGS Elevation Point Query Service
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    # define rest query params
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    # format query string and return query value
    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']

    return elevation

def unzip_file(filename, source_dir, destination_dir):
    """
    Unzip file, save results to another directory.

    :param filename: File to unzip, with full paths
    :param source_dir: Directory to pull zipped file from.
        This part of the path is removed from the beginning of filename to determine where to save unzipped files.
    :param destination_dir: Directory to save unzipped files to.
        In each file path, source_dir is replaced with destination_dir to determine where to save unzipped files.
    :returns: list of paths to unzipped files

    """
    # check that file is inside source_dir
    if os.path.samefile(source_dir, os.path.commonpath([filename, source_dir])):
        relfile = os.path.relpath(filename, source_dir)
        destination_file = os.path.join(destination_dir, os.path.splitext(relfile)[0])
        with zipfile.ZipFile(filename, 'r') as zf:
            zf.extractall(destination_file)
    else:
        raise FileNotFoundError(f'File {f} not in directory {source_dir}')
    return destination_file


def make_obs_interpolated(in_file, out_file, depths):
    """
    Add "interpolated_depth" column to obs file and save to new file.
    For now, this uses nearest neighbor interpolation.

    :param in_file: Filename to read observations from.
    :param out_file: Filename to save observations with extra column to.
    :param depths: Depths to interpolate to.

    """

    # Assign depth to observations (nearest neighbor)
    # Read temperature observations
    obs = pd.read_csv(in_file, parse_dates=['date'])
    # lake depths to use for LSTM
    depths = np.array(depths)
    # Round depth to nearest neighbor
    depth_interpolant = interpolate.interp1d(depths, depths,
                                             kind='nearest',
                                             assume_sorted=False)
    obs['interpolated_depth'] = depth_interpolant(obs.depth)
    obs.to_csv(out_file)


def make_lake_metadata_augmented(in_file, out_file):
    """
    Add "elevation" column to metadata file and save to new file

    :param in_file: filename to read lake metadata from
    :param out_file: filename to save to

    """
    lake_metadata = pd.read_csv(in_file)
    # Add elevation to metadata
    lake_metadata['elevation'] = lake_metadata.apply(
        lambda row: elevation_function(
            row['centroid_lat'], row['centroid_lon']),
        axis=1)
    lake_metadata.to_csv(out_file)


def unzip_all(in_files, log_file, source_dir, destination_dir):
    """
    Unzip all files in in_files

    :param in_files: List of files to unzip.
    :param log_file: File to log results of unzipping.
    :param source_dir: Directory to pull zipped files from.
        This part of the path is removed from the beginning of files to determine where to save unzipped files.
    :param destination_dir: Directory to save unzipped files to.
        In each file path, source_dir is replaced with destination_dir to determine where to save unzipped files.

    """

    # Create destination_dir if necessary
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # unzip files to 2_process/tmp folder
    zip_files = [f for f in in_files if f.lower().endswith('.zip')]
    unzipped_files = [unzip_file(zf, source_dir, destination_dir)
                      for zf in zip_files]
    with open(log_file, 'w') as f:
        f.write("Unzipping:\n\n")
        for zj in zip_files:
            f.write(zj + '\n')
        f.write("\nUnzipped these files:\n")
        for unzipped_file in unzipped_files:
            f.write(unzipped_file + '\n')

