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
    Query service using lat, lon. add the elevation values as a new column.
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


def all_dates_depths(df, depths, depth_col='interpolated_depth', pad_before_days=0, pad_after_days=0):
    """
    Fill NaNs for missing obs
    """
    
    pad_before_days_td = np.timedelta64(pad_before_days, 'D')
    pad_after_days_td = np.timedelta64(pad_after_days, 'D')
    # buffer start_date by pad_before_days days
    start_date = df.date.min() - pad_before_days_td
    # buffer end_date by pad_after_days days
    end_date = df.date.max() + pad_after_days_td
    date_range = pd.date_range(start=start_date, end=end_date)

    # create MultiIndex of all combinations of dates and depths
    midx = pd.MultiIndex.from_product([date_range, depths], names=['date', depth_col])
    # Average duplicate measurements (same day, same nearest depth)
    df_midx = df.groupby(['date', depth_col]).temp.mean().to_frame()
    df_timesteps = df_midx.reindex(midx, fill_value=np.NaN)

    # reshape to have one column per depth
    df_timesteps = df_timesteps.unstack()
    # flatten column levels
    df_timesteps.columns = [' '.join([str(s) for s in col]).strip() for col in df_timesteps.columns.values]
    # Now df_timesteps has dimensions of len(date_range) by len(depths)
    # values are NaN wherever there are no observations
    return df_timesteps


def assemble_data(lake_metadata, unzipped_dir, config):
    """
    Assemble raw data into equal-length sequences for training/testing

    :param lake_metadata: TODO
    :returns: TODO

    """
    # Read hyperparameters
    sequence_length = config['sequence_length']
    sequence_offset = config['sequence_offset']
    # pad time before 1st obs, e.g., 100 days
    spinup_time = config['spinup_time'] 

    # Read temperature observations
    obs = pd.read_csv(config['obs_file'], parse_dates=['date'])
    # lake depths to use for LSTM
    depths = np.array(config['depths'])
    # Round depth to nearest neighbor
    depth_interpolant = interpolate.interp1d(depths, depths,
                                             kind='nearest',
                                             assume_sorted=False)
    obs['interpolated_depth'] = depth_interpolant(obs.depth)

    # Add elevation to metadata
    lake_metadata['elevation'] = lake_metadata.apply(
        lambda row: elevation_function(
            row['centroid_lat'], row['centroid_lon']),
        axis=1)

    for lake in lake_metadata:
        # lake observations
        lake_obs = obs[obs.site_id==lake.site_id].copy()

        ### Fill NaNs for missing obs
        df_timesteps = all_dates_depths(lake_obs, depths, pad_before_days=spinup_time)

        '''
        spinup_time_td = np.timedelta64(spinup_time, 'D')
        # buffer start_date by spinup_time days
        start_date = lake_obs.date.min() - spinup_time_td
        end_date = lake_obs.date.max()
        date_range = pd.date_range(start=start_date, end=end_date)

        midx = pd.MultiIndex.from_product([date_range, depths], names=['date', 'interpolated_depth'])
        # Average duplicate measurements (same day, same nearest depth)
        lake_obs_midx = lake_obs.groupby(['date', 'interpolated_depth']).temp.mean().to_frame()

        df_timesteps = lake_obs_midx.reindex(midx, fill_value=np.NaN)
        # reshape to have one column per depth
        df_timesteps = df_timesteps.unstack()
        # flatten column levels
        df_timesteps.columns = [' '.join([str(s) for s in col]).strip() for col in df_timesteps.columns.values]
        '''
        # Now df_timesteps has dimensions of len(date_range) by len(depths)
        # values are NaN wherever there are no observations

        ### Join drivers

        #2_process/tmp/drivers_mntoha/inputs_01_N48.00-49.50_W89.50-97.25/nldas_meteo_N48.0625-48.0625_W90.6875-90.6875.csv
        drivers_file = f'{destination_dir}/drivers_mntoha/inputs_{lake.group_id}/{lake.meteo_filename}'
        drivers = pd.read_csv(drivers_file, parse_dates=['time'], index_col='time')
        # TODO: Check for nans
        # drivers.isna().any()
        df_timesteps = df_timesteps.join(drivers, how='left')

        ### Join clarity
        clarity_file = f'{destination_dir}/drivers_mntoha/clarity_{lake.group_id}/gam_{lake.site_id}_clarity.csv'
        clarity = pd.read_csv(clarity_file, parse_dates=['date'], index_col='date')
        df_timesteps = df_timesteps.join(clarity, how='left')

        ### Join ice flags
        ice_flags_file = f'{destination_dir}/drivers_mntoha/ice_flags_{lake.group_id}/pb0_{lake.site_id}_ice_flags.csv'
        ice_flags = pd.read_csv(ice_flags_file, parse_dates=['date'], index_col='date')
        df_timesteps = df_timesteps.join(ice_flags, how='left')

        ## Get attributes
        df_timesteps['area'] = lake['area']
        df_timesteps['lon'] = lake['centroid_lon']
        df_timesteps['lat'] = lake['centroid_lat']
        df_timesteps['elevation'] = lake['elevation']
        # Now df_timesteps is one long timeseries with depth-specific observations, drivers, and attributes as columns

        ## Split into segments

        # Let's make some rules:
        # 1. 400 day sequences
        # 2. First 100 days are spinup, so don't have observations then
        # 3. Each next sequence starts 200 days later than the previous
        # 4. Discard sequences that lack observations
        # 5. We know going in that the first and last sequence have observations

        # Convert dataframe to numpy array
        arr_timesteps = df_timesteps.to_numpy(dtype=np.float32)
        # Create a strided view into arr_timesteps
        # This creates a numpy object with sequences of length sequence_length
        # and offsets of sequence_offset without making a copy the array
        all_sequences = np.lib.stride_tricks.sliding_window_view(
            arr_timesteps, sequence_length, axis=0
        )[::sequence_offset, :]
        # shape is now (# sequences, # features + depths, sequence_length)
        # The first `len(depths)` elements in the middle dimension are observations

        # Mask sequences that have only NaN observations
        mask_nan_sequences = np.any(np.any(~np.isnan(all_sequences[:, :len(depths), :]), axis=2), axis=1)
        # sequences to keep
        lake_sequences = all_sequences[mask_nan_sequences, :, :].transpose((0, 2, 1))

        # also keep the last `sequence_length` days, since the final observation is always non-NaN,
        # unless all_sequences already has that last sequence 
        if (arr_timesteps.shape[0] - sequence_length) % sequence_offset != 0:
            lake_sequences = np.concatenate((lake_sequences, np.array([arr_timesteps[-sequence_length:, :]])), axis=0)

        sequences_file = os.path.join('2_process', 'out', f'sequences_{lake.site_id}.npy')
        np.save(sequences_file, lake_sequences, allow_pickle=True)


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

