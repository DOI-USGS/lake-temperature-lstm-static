import os
import zipfile
import requests
import urllib

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

    :param source_dir: Directory to pull zipped files from.
        This part of the path is removed from the beginning of files to determine where to save unzipped files.
    :param in_files: 
    :param log_file: 
    :param destination_dir: 

    """

    # Create destination_dir if necessary
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # unzip files to 2_process/tmp folder
    zip_files = [f for f in in_files if f.lower().endswith('.zip')]
    with open(log_file, 'w') as f:
        f.write("Unzipping:\n\n")
        for zip_file in zip_files:
            f.write(zip_file + '\n')
    unzipped_files = unzip_files(zip_files, source_dir, destination_dir)
    with open(log_file, 'a') as f: 
        f.write("\nUnzipped these files:\n")
        for unzipped_file in unzipped_files:
            f.write(unzipped_file + '\n')

