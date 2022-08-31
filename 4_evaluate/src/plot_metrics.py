import numpy as np
import matplotlib.pyplot as plt

'''
TODO: 
- Get observations from unnormalized outputs
- Ignore spinup time
- Write one plotting function per plot
- Write metric-computing functions
  - training "bias" by depth
  - training std by depth
  - training "bias" by doy
  - training std by doy
  - RMSE by depth
  - RMSE by doy
  - Bias by depth
  - Bias by doy
- Call each from plot_all_metrics
- Only necessary metrics stored in scope of plot_all_metrics
- Keep full data/obs out of plot_all_metrics scope
'''


def get_rmse_lake(metadata_filepath,
                  normalized_dataset_filepath,
                  unnormalized_data_set_filepath,
                  predictions_filepath):
    """
    Compute RMS error at each lake over a dataset (training, validation, or
    testing).

    :param predictions_filepath: 

    """

    # Load metadata to get spinup time
    metadata_filepath = "3_train/out/model_prep/initial/local_a_metadata.npz"
    metadata_npz = np.load(metadata_filepath)
    # Convert npz to dictionary
    # Convert arrays to scalars where possible
    metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}
    spinup_time = metadata['spinup_time']

    # Load normalized observations to get site ids
    normalized_dataset_filepath = "2_process/out/model_prep/valid.npz"
    normalized_dataset_npz = np.load(normalized_dataset_filepath)
    site_ids = normalized_dataset_npz['site_ids']

    # Load unnormalized observations to get observations
    unnormalized_dataset_filepath = "4_evaluate/out/model_prep/initial/local_a/unnormalized_valid.npz"
    unnormalized_dataset_npz = np.load(unnormalized_dataset_filepath)

    # Load predictions
    predictions_filepath = "4_evaluate/out/model_prep/initial/local_a/preds_valid.npz"
    predictions_npz = np.load(predictions_filepath)

    # Where are valid observations in time and space
    i_obs = ~np.isnan(unnormalized_dataset_npz['observations'][:, spinup_time:, :])
    # Number of observations per sequence
    obs_counts = np.sum(i_obs, axis=(1,2))
    # Squared Error per sequence
    seq_SE = np.nansum((unnormalized_dataset_npz['observations'][:, spinup_time:, :] -
                        predictions_npz['predictions'])**2, axis=(1,2))
    unique_site_ids = np.unique(site_ids)
    lake_RMSE = {}
    for unique_site_id in unique_site_ids:
        # Index of every sequence for this site
        i_sequences = np.where(site_ids == unique_site_id)
        # Total # observations for this site
        n_obs = sum(obs_counts[i_sequences])
        # Sum all squared errors for this site
        lake_SE = sum(seq_SE[i_sequences])
        # RMSE for this site
        lake_RMSE[unique_site_id] = np.sqrt(lake_SE/n_obs)

    return lake_RMSE


def get_bias_lake(predictions_filepath):
    """
    Compute bias at each lake over a dataset (training, validation, or
    testing).
    """

    # Load metadata to get spinup time
    metadata_filepath = "3_train/out/model_prep/initial/local_a_metadata.npz"
    metadata_npz = np.load(metadata_filepath)
    # Convert npz to dictionary
    # Convert arrays to scalars where possible
    metadata = {file: metadata_npz[file][()] for file in metadata_npz.files}
    spinup_time = metadata['spinup_time']

    # Load normalized observations to get site ids
    normalized_dataset_filepath = "2_process/out/model_prep/valid.npz"
    normalized_dataset_npz = np.load(normalized_dataset_filepath)
    site_ids = normalized_dataset_npz['site_ids']

    # Load unnormalized observations to get observations
    unnormalized_dataset_filepath = "4_evaluate/out/model_prep/initial/local_a/unnormalized_valid.npz"
    unnormalized_dataset_npz = np.load(unnormalized_dataset_filepath)

    # Load predictions
    predictions_filepath = "4_evaluate/out/model_prep/initial/local_a/preds_valid.npz"
    predictions_npz = np.load(predictions_filepath)

    # Where are valid observations in time and space
    i_obs = ~np.isnan(unnormalized_dataset_npz['observations'][:, spinup_time:, :])
    # Number of observations per sequence
    obs_counts = np.sum(i_obs, axis=(1,2))
    # Squared Error per sequence
    seq_SE = np.nansum((unnormalized_dataset_npz['observations'][:, spinup_time:, :] -
                        predictions_npz['predictions'])**2, axis=(1,2))
    unique_site_ids = np.unique(site_ids)
    lake_RMSE = {}
    for unique_site_id in unique_site_ids:
        # Index of every sequence for this site
        i_sequences = np.where(site_ids == unique_site_id)
        # Total # observations for this site
        n_obs = sum(obs_counts[i_sequences])
        # Sum all squared errors for this site
        lake_SE = sum(seq_SE[i_sequences])
        # RMSE for this site
        lake_RMSE[unique_site_id] = np.sqrt(lake_SE/n_obs)

    return lake_RMSE

    pass


def get_rmse_depth(predictions_filepath):
    """
    Compute RMS error at each output depth over a dataset (training,
    validation, or testing).
    """

    # Load predictions
    # TODO: include try/catch for if the file doesn't exist?
    predictions_npz = np.load(predictions_filepath)
    residuals = predictions_npz['observations'] - predictions_npz['predictions']
    # Where are valid observations in time and space
    iobs = ~np.isnan(residuals)
    # Number of observations per depth
    obs_counts = np.sum(iobs, axis=(0,1))
    # Squared Error per depth
    seq_SE = np.nansum((residuals)**2, axis=(0,1))
    rmse_by_depth = np.sqrt(seq_SE/obs_counts)
    return rmse_by_depth


def get_bias_depth(predictions_filepath):
    """
    Compute bias at each output depth over a dataset (training, validation, or
    testing).
    """

    # Load predictions
    predictions_filepath = "4_evaluate/out/model_prep/initial/local_a/preds_valid.npz"
    predictions_npz = np.load(predictions_filepath)
    bias_by_depth = np.nanmean((predictions_npz['observations'] - predictions_npz['predictions']), axis=(0,1))
    return bias_by_depth


def get_rmse_doy(predictions_filepath, metadata_filepath):
    """
    

    :param predictions_filepath: 
    :param metadata_filepath: 

    """
    # Load predictions
    # TODO: include try/catch for if the file doesn't exist?
    predictions_npz = np.load(predictions_filepath)
    # TODO: Load starts from original dataset (foolproof)
    # NOTE: While validation start dates are in the metadata,
    # test start dates have to be loaded from 2_process/out/model_prep/test.npz
    # Load metadata
    metadata = np.load(metadata_filepath)
    starts = metadata['valid_start_dates']
    # get indices of every non-NaN residual
    i0s, i1s, i2s = np.where(~np.isnan(predictions_npz['observations']))
    # get non_NaN residuals
    non_nan_residuals = predictions_npz['observations'][i0s, i1s, i2s] - predictions_npz['predictions'][i0s, i1s, i2s]
    # get day-of-year of every non-NaN residual
    res_day_of_year = np.array([(starts[i0] + i1).astype(datetime).timetuple().tm_yday for i0, i1 in zip(i0s, i1s)])
    days_of_year = np.arange(1, 367)
    rmse_by_doy = np.array([np.sqrt(np.mean(non_nan_residuals[res_day_of_year==doy]**2)) for doy in days_of_year])
    return rmse_by_doy


def get_bias_doy(predictions_filepath, metadata_filepath):
    """
    """
    # Load predictions
    # TODO: include try/catch for if the file doesn't exist?
    predictions_npz = np.load(predictions_filepath)
    # Load metadata
    metadata = np.load(metadata_filepath)
    # get indices of every non-NaN residual
    i0s, i1s, i2s = np.where(~np.isnan(predictions_npz['observations']))
    # get non_NaN residuals
    non_nan_residuals = predictions_npz['observations'][i0s, i1s, i2s] - predictions_npz['predictions'][i0s, i1s, i2s]
    # get day-of-year of every non-NaN residual
    res_day_of_year = np.array([(starts[i0] + i1).astype(datetime).timetuple().tm_yday for i0, i1 in zip(i0s, i1s)])
    days_of_year = np.arange(1, 367)

    bias_by_doy = np.array([np.mean(non_nan_residuals[res_day_of_year==doy]) for doy in days_of_year])
    return bias_by_doy


def get_metric(metric_name, by, predictions_filepath, metadata_filepath=None):
    """
    Compute a metric (mean, RMSE, or bias) for a dataset (training, validation,
    or testing) by day of year, depth, or by lake.

    :param metric_name: 
    :param by: 
    :param predictions_filepath: 
    :param metadata_filepath:  (Default value = None)

    """
    bad_by_message = 'The argument "by" must be: "depth", "doy", or "lake"'
    bad_metric_name_message = 'The argument "metric_name" must be: "rmse" or "bias"'
    if metric_name.lower() == 'rmse':
        if by.lower() == 'depth':
            metric = get_rmse_depth(predictions_filepath)
        elif by.lower() == 'doy':
            metric = get_rmse_doy(predictions_filepath, metadata_filepath)
        elif by.lower() == 'lake':
            metric = get_rmse_lake(predictions_filepath)
        else:
            raise ValueError(bad_by_message)
    elif metric_name.lower() == 'bias':
        if by.lower() == 'depth':
            metric = get_bias_depth(predictions_filepath)
        elif by.lower() == 'doy':
            metric = get_bias_doy(predictions_filepath, metadata_filepath)
        elif by.lower() == 'lake':
            metric = get_bias_lake(predictions_filepath)
        else:
            raise ValueError(bad_by_message)
    else:
        raise ValueError(bad_metric_name_message)

    return metric

def plot_training_bias_depth():
    pass

def plot_training_std_depth():
    pass

def plot_training_bias_doy():
    pass

def plot_training_std_doy():
    pass

def plot_rmse_depth(predictions_filepath, metadata_filepath, plot_filepath):
    """
    Plot RMS Error as a function of depth for a dataset (training, validation,
    or testing). Save resulting plot to file.

    :param predictions_filepath: 
    :param metadata_filepath: 
    :param plot_filepath: 

    """
    rmse_by_depth = get_rmse_depth(predictions_filepath)
    metadata = np.load(metadata_filepath)
    depths = metadata['depths_use']

    # How does RMSE vary by depth?
    plt.plot(depths, rmse_by_depth)
    plt.xlabel('Depth, m')
    plt.ylabel('RMSE, °C')
    plt.savefig(plot_filepath)

def plot_rmse_doy():
    pass

def plot_bias_depth():
    pass

def plot_bias_doy():
    pass

def plot_lake_evaluation():
    pass

def plot_all_metrics():
    pass

def plot_metric():
    pass

if __name__ == '__main__':
    plot_all_metrics(snakemake.input.weights_filepath,
                     snakemake.input.metadata_filepath,
                     snakemake.input.dataset_filepath,
                     snakemake.output.predictions_filepath)


