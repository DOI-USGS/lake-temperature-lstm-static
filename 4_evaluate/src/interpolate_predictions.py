import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Evaluate predictions the same way as lake-temperature-process-models:
Interpolate ML output to observation depths.
"""

def interpolate_predictions(predictions_filepath,
                            sequence_indices,
                            date_indices,
                            depth_above_indices,
                            depth_below_indices,
                            above_factors,
                            below_factors,
                            use_obs):
    """
    Load and interpolate predictions from predicted depths to observation depths

    :param predictions_filepath: Path to saved unnormalized predictions
    :param sequence_indices: Index of every observation's corresponding sequence in predictions array
    :param date_indices: Index for every observation within its sequence
    :param depth_above_indices: Indexing array for the predictions above the observation locations
    :param depth_below_indices: Indexing array for the predictions below the observation locations
    :param above_factors: Interpolation factors for the predictions above the observations
    :param below_factors: Interpolation factors for the predictions below the observations
    :param use_obs: Indexing array for observation locations to interpolate to
    :returns: Array of temperature predictions interpolated to observation depths

    """
    # Load predictions
    predictions_npz = np.load(predictions_filepath)
    ta = predictions_npz['predictions'][sequence_indices[use_obs], date_indices[use_obs], depth_above_indices[use_obs]]
    tb = predictions_npz['predictions'][sequence_indices[use_obs], date_indices[use_obs], depth_below_indices[use_obs]]
    # Interpolate
    predictions = ta*above_factors[use_obs] + tb*below_factors[use_obs]
    return predictions


def get_interpolated_predictions(predictions_filepath,
                                 metadata_filepath,
                                 dataset_filepath,
                                 obs_file,
                                 interpolated_predictions_filepath,
                                 batch_size=5000):
    """
    Interpolate ML output to observation depths.

    :param predictions_filepath: Path to unnormalized predictions for one
        dataset (train, validation, or test)
    :param metadata_filepath: Path to model training metadata npz for one dataset
    :param dataset_filepath: Path to input for one dataset
    :param obs_file: File with all temperature observations
    :param interpolated_predictions_filepath: csv file to save interpolated
        predictions to
    :param batch_size: Number of observations to use per batch while matching
        observations to prediction sequences (Default value = 5000)

    """

    # Load observations... from original data!
    # Load normalized observations to get site ids
    dataset_npz = np.load(dataset_filepath)
    site_ids = dataset_npz['site_ids']
    start_dates = dataset_npz['start_dates']
    sequence_length = dataset_npz['sequence_length']
    spinup_time = dataset_npz['spinup_time']
    sequence_offset = dataset_npz['sequence_offset']
    depths_all = dataset_npz['depths_all']
    dynamic_features_all = dataset_npz['dynamic_features_all']
    static_features_all = dataset_npz['static_features_all']

    # Load metadata to get depths_use
    metadata_npz = np.load(metadata_filepath)
    depths_use = metadata_npz['depths_use']

    # Now, we'll need obs for each of those lakes.
    # Read temperature observations
    obs_all = pd.read_csv(obs_file, index_col=0, parse_dates=['date'])
    # Whittle down to lakes in this dataset
    obs = obs_all[obs_all.site_id.isin(site_ids)]
    obs_all.site_id.unique().shape, obs.site_id.unique().shape, np.unique(site_ids).shape

    # Prepare to find the indices of the predictions that are needed to
    # interpolate temperatures to observation depths, for every observation in
    # the dataset. We need sequence index, index of day within sequence, and
    # two depth indices: one above the observation, one below. First let's get
    # sequence index and date index.
    pred_starts = start_dates + spinup_time
    pred_ends = start_dates + sequence_length
    n_obs = len(obs)
    obs_sites = obs.site_id.to_numpy('str')
    obs_dates_days = obs.date.to_numpy(dtype='datetime64[D]').astype(int)
    pred_starts_days = pred_starts.astype(int)
    pred_ends_days = pred_ends.astype(int)

    # Find prediction indices for every observation, looping over batches of
    # observations. Without batches, this could require many GBs of memory: One
    # byte per bool (that's how numpy store bool arrays) Batches to avoid
    # memory burden while allowing some vectorized ops for speedup
    n_batches = int(np.ceil(n_obs/batch_size))
    sequence_indices = np.empty(n_obs, int)
    date_indices = np.empty(n_obs, int)
    for i_batch in tqdm(range(n_batches)):
        batch_start = i_batch*batch_size
        batch_end = min((i_batch + 1)*batch_size, n_obs)
        this_batch_size = batch_end-batch_start

        # 3 conditions: right site ID, after sequence starts, and before sequence ends
        # broadcast for speedup
        idx_obs, idx_sequence = np.nonzero(
            (obs_sites[batch_start:batch_end, np.newaxis] == site_ids) &
            (obs_dates_days[batch_start:batch_end, np.newaxis] >= pred_starts_days) &
            (obs_dates_days[batch_start:batch_end, np.newaxis] < pred_ends_days)
        )
        # Check to see if any sequences were found for this batch
        if idx_obs.size == 0:
            sequence_indices[batch_start:batch_end] = -1
            date_indices[batch_start:batch_end] = -1
        else:
            # Now select the sequence with the earliest start date
            batch_start_dates = pred_starts_days[idx_sequence]
            start_date_matrix = np.where(
                idx_obs[:, np.newaxis] == np.arange(this_batch_size),
                batch_start_dates[:, np.newaxis],
                np.NaN
            )
            # Make sure that no column of start_date_matrix is all NaN
            # Otherwise, nanargmin will error
            no_nans = ~np.isnan(np.nanmin(start_date_matrix, axis=0))
            which_sequence = np.nanargmin(start_date_matrix[:, no_nans], axis=0)
            sequence_indices_batch = np.array([-1]*this_batch_size)
            sequence_indices_batch[no_nans] = idx_sequence[which_sequence]
            # Now, get the index of the date within the sequence for each obs
            date_indices_batch = np.array([-1]*this_batch_size)
            date_indices_batch[no_nans] = (obs_dates_days[batch_start:batch_end][no_nans] - 
                                           pred_starts_days[sequence_indices_batch[no_nans]])
            sequence_indices[batch_start:batch_end] = sequence_indices_batch
            date_indices[batch_start:batch_end] = date_indices_batch

    # Now, get the prediction depth above and below each observation
    obs_depths = obs.depth.to_numpy()
    # Clip in order to extrapolate outside predicted depth range
    clipped_obs_depths = np.clip(obs_depths, min(depths_use), max(depths_use))
    # Pad depths with one value that is always below obs,
    # for convenience in interpolation algorithm below
    pad_value = max(np.r_[depths_all, clipped_obs_depths] + 1)
    padded_pred_depths = np.r_[depths_all, pad_value]

    # Get index and value of next largest depth in list of depths
    def enumerate_depth_below(depth):
        return next((ii, x) for ii, x in enumerate(padded_pred_depths) if x > depth)
    depth_below_tuples = [enumerate_depth_below(obs_depth) for obs_depth in clipped_obs_depths]

    # Unpack list of two-element tuples into two arrays
    depth_below_index, depth_below = (np.array(tp) for tp in zip(*depth_below_tuples))
    depth_above_index = depth_below_index - 1
    depth_above = depths_all[depth_above_index]

    # Compute interpolation factors
    above_factors = (depth_below - clipped_obs_depths)/(depth_below - depth_above)
    below_factors = (clipped_obs_depths - depth_above)/(depth_below - depth_above)
    # Check that the factors sum to 1
    assert all(above_factors + below_factors == 1)

    # Now, some depth_below_index values are one greater than the maximum index in depths_use
    # The corresponding below_factors should be zero, though
    # Let's double check that
    too_low = np.nonzero(depth_below_index==len(depths_use))
    assert all(below_factors[too_low] == 0)
    # Good. That means that the depth_below_index doesn't need to be used when it's one greater than max
    # because those values get multiplied by zero
    # Set those indices to the max index instead
    depth_below_index[too_low] = len(depths_use) - 1

    # Let's now skip any obs with indices equal to -1
    # TODO: Consider dropping observations that are deeper than deepest depth_use
    use_obs = ((sequence_indices >= 0) & (date_indices >= 0))

    # Now the tricky part (like it hasn't been tricky so far!):
    # Access the temperatures in the predictions array

    # Interpolate predictions
    predictions = interpolate_predictions(predictions_filepath,
                                          sequence_indices,
                                          date_indices,
                                          depth_above_index,
                                          depth_below_index,
                                          above_factors,
                                          below_factors,
                                          use_obs)

    obs_preds = obs.assign(pred=np.NaN)
    obs_preds.loc[use_obs, 'pred'] = predictions
    obs_preds.to_csv(interpolated_predictions_filepath, index=False)


if __name__ == '__main__':
    get_interpolated_predictions(snakemake.input.predictions_filepath,
                                 snakemake.input.metadata_filepath,
                                 snakemake.input.dataset_filepath,
                                 snakemake.input.obs_file,
                                 snakemake.output.interpolated_predictions_filepath,
                                 snakemake.params.batch_size)

