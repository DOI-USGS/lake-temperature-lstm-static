import numpy as np
import pandas as pd

def all_dates_depths(lake_obs_interpolated, depths, depth_col='interpolated_depth', pad_before_days=0, pad_after_days=0, min_days=1):
    """
    Create a DataFrame with one column per depth and one row for every day from
    `pad_before_days` days before the first observation until `pad_after_days`
    after the last observation. Missing observations are filled in with np.NaN.

    :param lake_obs_interpolated: DataFrame of observations for one lake, interpolated to the depths in the list `depths`.
    :param depths: Full list of depths at which temperatures will be provided by the trained model.
    :param depth_col: Name of DataFrame column with depth values to match up with `depths`. (Default value = 'interpolated_depth')
    :param pad_before_days: Number of days to pad before the first observation (Default value = 0)
    :param pad_after_days: Number of days to pad after the last observation (Default value = 0)
    :param min_days: Minimum number of days included in output DataFrame (Default value = 1)
    :returns: DataFrame of observations with NaN values for missing observations.

    """

    pad_before_days_td = np.timedelta64(pad_before_days, 'D')
    pad_after_days_td = np.timedelta64(pad_after_days, 'D')
    min_days_td = np.timedelta64(min_days, 'D')
    one_day = np.timedelta64(1, 'D')
    # buffer start_date by pad_before_days days
    start_date = lake_obs_interpolated.date.min() - pad_before_days_td
    # buffer end_date by pad_after_days days
    end_date = lake_obs_interpolated.date.max() + pad_after_days_td
    # check if enough days are included, pad if not
    if end_date - start_date + one_day < min_days_td:
        start_date = end_date - min_days_td + one_day
    date_range = pd.date_range(start=start_date, end=end_date)

    # create MultiIndex of all combinations of dates and depths
    midx = pd.MultiIndex.from_product([date_range, depths], names=['date', depth_col])
    # Average duplicate measurements (same day, same nearest depth)
    obs_midx = lake_obs_interpolated.groupby(['date', depth_col]).temp.mean().to_frame()
    obs_all = obs_midx.reindex(midx, fill_value=np.NaN)

    # reshape to have one column per depth
    obs_all = obs_all.unstack()
    # flatten column levels
    obs_all.columns = [' '.join([str(s) for s in col]).strip() for col in obs_all.columns.values]
    # Now obs_all has dimensions of len(date_range) by len(depths)
    # values are NaN wherever there are no observations
    return obs_all


def assemble_lake_data(site_id,
                       metadata_augmented_file,
                       obs_interpolated_file, 
                       drivers_file,
                       clarity_file,
                       ice_flags_file,
                       config):
    """
    Assemble raw data from one lake into equal-length sequences for
    training/testing.

    :param site_id: site_id of lake to process
    :param metadata_augmented_file: lake metadata file, augmented with elevation
    :param obs_interpolated_file: temperature observations file, interpolated
        to LSTM depths
    :param drivers_file: Meteorological drivers csv for this lake
    :param clarity_file: Time-varying lake clarity csv
    :param ice_flags_file: Time-varying ice flags csv for this lake
    :param config: Snakemake config for process_mntoha
    :returns: Numpy array of sequences with shape (# sequences,
        sequence_length, # features + depths)

    """
    # Read hyperparameters
    sequence_length = config['sequence_length']
    sequence_offset = config['sequence_offset']
    # pad time before 1st obs, e.g., 100 days
    spinup_time = config['spinup_time'] 
    # Discretized lake depths to use for LSTM
    depths = np.array(config['depths'])

    # Read temperature observations
    obs_interpolated = pd.read_csv(obs_interpolated_file, parse_dates=['date'])
    # lake observations
    lake_obs_interpolated = obs_interpolated[obs_interpolated.site_id==site_id].copy()
    # determine if there are any observations
    if len(lake_obs_interpolated) == 0:
        lake_sequences = np.empty((0))
    else:
        # Read metadata with elevation
        lake_metadata_augmented = pd.read_csv(metadata_augmented_file)
        # NOTE: If there are duplicates, the next line will only grap the first.
        # If there are no matches, it will throw an error
        lake = lake_metadata_augmented[lake_metadata_augmented.site_id==site_id].iloc[0,:]

        # Fill NaNs for missing obs
        obs_all = all_dates_depths(lake_obs_interpolated,
                                   depths,
                                   pad_before_days=spinup_time,
                                   min_days=sequence_length)
        # Now obs_all has dimensions of len(date_range) by len(depths)
        # values are NaN wherever there are no observations

        # Join drivers

        drivers = pd.read_csv(drivers_file, parse_dates=['time'], index_col='time')
        # TODO: Check for nans
        # drivers.isna().any()
        obs_all = obs_all.join(drivers, how='left')

        # Join clarity
        clarity = pd.read_csv(clarity_file, parse_dates=['date'], index_col='date')
        obs_all = obs_all.join(clarity, how='left')

        # Join ice flags
        ice_flags = pd.read_csv(ice_flags_file, parse_dates=['date'], index_col='date')
        obs_all = obs_all.join(ice_flags, how='left')

        # Get attributes
        obs_all['area'] = lake['area']
        obs_all['lon'] = lake['centroid_lon']
        obs_all['lat'] = lake['centroid_lat']
        obs_all['elevation'] = lake['elevation']
        # Now obs_all is one long timeseries with depth-specific observations,
        # drivers, clarity, ice_flags, and attributes as columns

        # Split into segments

        # Let's make some rules for segments:
        # 1. sequences are sequence_length days long
        # 2. First spinup_time days are spinup, so don't have observations then
        # 3. Each next sequence starts sequence_offset days later than the previous
        # 4. Discard sequences that lack observations
        # 5. We know going in that the first and last sequence have observations

        # Convert dataframe to numpy array
        obs_all_array = obs_all.to_numpy(dtype=np.float32)
        # Create a strided view into obs_all_array
        # This creates a numpy object with sequences of length sequence_length
        # and offsets of sequence_offset without making a copy the array
        all_sequences = np.lib.stride_tricks.sliding_window_view(
            obs_all_array, sequence_length, axis=0
        )[::sequence_offset, :]
        # shape is now (# sequences, # features + depths, sequence_length)
        # The first `len(depths)` elements in the middle dimension are observations

        # Mask sequences that have only NaN observations
        mask_nan_sequences = np.any(np.any(~np.isnan(all_sequences[:, :len(depths), :]), axis=2), axis=1)
        # sequences to keep
        lake_sequences = all_sequences[mask_nan_sequences, :, :].transpose((0, 2, 1))
        # Now, shape is (# sequences, sequence_length, # features + depths)

        # also keep the last `sequence_length` days, since the final observation is always non-NaN,
        # unless all_sequences already has that last sequence 
        if (obs_all_array.shape[0] - sequence_length) % sequence_offset != 0:
            lake_sequences = np.concatenate((lake_sequences, np.array([obs_all_array[-sequence_length:, :]])), axis=0)

    return lake_sequences


def main(site_id,
         out_file,
         metadata_augmented_file,
         obs_interpolated_file,
         drivers_file,
         clarity_file,
         ice_flags_file,
         config):
    """
    Process raw data for one MNTOHA lake
    Save those data to a numpy file

    :param site_id: site_id of lake to process
    :param out_file: Filename of npy file to save
    :param metadata_augmented_file: lake metadata file, augmented with elevation
    :param obs_interpolated_file: temperature observations file, interpolated
        to LSTM depths
    :param drivers_file: Meteorological drivers csv for this lake
    :param clarity_file: Time-varying lake clarity csv
    :param ice_flags_file: Time-varying ice flags csv for this lake
    :param config: Snakemake config for process_mntoha

    """
    lake_sequences = assemble_lake_data(
        site_id,
        metadata_augmented_file,
        obs_interpolated_file,
        drivers_file,
        clarity_file,
        ice_flags_file,
        config
    )
    np.save(out_file, lake_sequences, allow_pickle=True)


if __name__ == '__main__':
    main(snakemake.wildcards['site_id'],
         snakemake.output[0], # npy file
         snakemake.input[0], # metadata_augmented_file
         snakemake.input[1], # obs_interpolated_file
         snakemake.input[2], # drivers_file
         snakemake.input[3], # clarity_file
         snakemake.input[4], # ice_flags_file
         snakemake.params['config'])

