import os
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import hvplot.pandas
import holoviews as hv


def load_predictions(predictions_filepath, doy_bin_width):
    """
    Load predictions for one dataset from a csv as a pandas Dataframe.
    Load the predictions for the training, validation, or testing set. Remove
    any NaN predictions. Compute residuals (as observed - predicted) and save
    them as an extra column in the dataframe.

    :param predictions_filepath: Path to predictions csv
    :param doy_bin_width: Number of days to bin day of year to. For example, a
        doy_bin_width of 7 means that the values in the column doy_bin will
        increase by 7 at a time. Useful for smoothing out plots of metrics by
        day of year.

    """
    raw_preds = pd.read_csv(predictions_filepath, parse_dates=['date'])
    # remove nans
    non_nan_preds = raw_preds.loc[~raw_preds['predicted_temperature_obs_depth'].isna()].copy()
    # Compute residuals (observed - predicted)
    non_nan_preds['residual'] = non_nan_preds['temp'] - non_nan_preds['predicted_temperature_obs_depth']
    # Add a column for day of year
    non_nan_preds['doy'] = non_nan_preds['date'].dt.day_of_year
    # Add a column for binned day of year
    non_nan_preds['doy_bin'] = (doy_bin_width*round((non_nan_preds['doy'].astype(float))/doy_bin_width)).astype(int)
    return non_nan_preds


def rms(df):
    """
    Return root mean squared of a Pandas Series or Dataframe. Useful as
    argument to groupby().aggregate().

    :param df: Pandas Series or Dataframe to compute RMS for

    """
    return np.sqrt((df**2).mean())


def plot_by_lake(agg_metric, lake_metadata_filepath, metric):
    """
    Create a scatter plot of modeled lakes with color indicating RMS error.

    :param agg_metric: Pandas Dataframe with aggregated metrics for each lake
    :param lake_metadata_filepath: Path to lake metadata csv
    :param metric: Metric to plot, 'rmse' or 'bias'. Used to determine colormap
    for points (diverging for bias, sequential for rmse)

    """
    # Augment with lake metadata
    lake_metadata = pd.read_csv(lake_metadata_filepath, index_col=0)
    df_plot = pd.merge(agg_metric, lake_metadata, how='left', on='site_id')
    cmaps = {
        'rmse': 'plasma',
        'bias': 'RdBu'
    }
    p = df_plot.hvplot.points(
        x='longitude',
        y='latitude',
        color=f'LSTM_{metric}',
        cmap=cmaps[metric],
        logz=False,
        hover_cols='all',
        geo=True,
        tiles='CartoLight',
        height=700,
        width=700
    )
    return p


def plot_metric(plot_filepath, predictions_filepath, metric, plot_by,
                lake_metadata_filepath=None,
                train_predictions_filepath=None,
                include_train_mean=False,
                doy_bin_width=1):
    """
    Plot a metric (e.g. RMS error) as a function of some variable (e.g. lake depth).
    
    This function allows plotting RMSE or bias by lake, by depth in the lake, or by day of year.
    
    There is also the option to include metrics for the mean of the training
    set. This allows performance to be compared against the variance of the
    observations, which is useful. For instance, RMSE is lower in winter, but
    that lower RMSE comes about because lake temperatures vary less during
    winter months, not by any virtue of the model. This is clear when the mean
    of the training data is included in the plots. Note that the RMSE of the
    mean of the training data is the standard deviation of the training data.

    :param plot_filepath: Path to save plot image to. Image format is
        determined from extension.
    :param predictions_filepath: Path to predictions csv
    :param metric: Metric to plot: 'rmse' or 'bias'
    :param plot_by: Independent variable for evaluating metric: 'lake',
        'depth', or 'doy'
    :param lake_metadata_filepath: Path to lake metadata csv (Default value =
        None). Only needed for plotting by lake, to get lake latitude and
        longitude.
    :param train_predictions_filepath: Path to training set predictions csv (Default value = None)
    :param include_train_mean: Whether to include metrics for the mean of the
        training set in the plot (Default value = False).
    :param doy_bin_width: Number of days to bin day of year to (Default value =
        1). For example, a doy_bin_width of 7 means that the values in the
        column doy_bin will increase by 7 at a time. Useful for smoothing out
        plots of metrics by day of year.

    """
    # Dictionary relating inputs to column names
    prediction_columns = {
        'lake': 'site_id',
        'depth': 'interpolated_depth',
        'doy': 'doy_bin'
    }
    # Dictionary relating inputs to aggregating functions
    agg_functions = {
        'rmse': rms,
        'bias': 'mean'
    }

    preds = load_predictions(predictions_filepath, doy_bin_width)

    # Aggregate metric of residuals by a variable
    agg_metric = (
        preds.groupby(prediction_columns[plot_by])['residual']
        .aggregate(agg_functions[metric])
        .rename(f'LSTM_{metric}')
    )

    if plot_by == 'lake':
        # Plotting by lake creates a geographic plot, so call separate function
        p = plot_by_lake(agg_metric, lake_metadata_filepath, metric)
    else:
        if include_train_mean:
            # For comparison, take the mean of the training set and use it as a predictor to get residuals
            # This shows visually where prediction is more challenging due to observation variance
            # or difference from the training data
            train_preds = load_predictions(train_predictions_filepath, doy_bin_width)
            train_mean_temp = train_preds.groupby(prediction_columns[plot_by])['temp'].mean().rename('train_mean_temp')
            preds_train_mean = preds.merge(train_mean_temp, how='left', on=prediction_columns[plot_by]).copy()
            preds_train_mean['train_mean_residual'] = preds_train_mean['temp'] - preds_train_mean['train_mean_temp']
            metric_train_mean = (
                preds_train_mean.groupby(prediction_columns[plot_by])['train_mean_residual']
                .aggregate(agg_functions[metric])
                .rename(f'train_mean_{metric}')
            )
            df_plot = pd.merge(agg_metric, metric_train_mean, how='outer', left_index=True, right_index=True)
        else:
            df_plot = agg_metric

        zero_line = hv.HLine(0).opts(color='black', level='underlay')
        if plot_by == 'depth':
            if metric == 'bias':
                p = df_plot.hvplot.line() * zero_line
            else:
                p = df_plot.hvplot.line()
        elif plot_by == 'doy':
            if metric == 'bias':
                p = df_plot.hvplot.line() * zero_line
            else:
                p = df_plot.hvplot.line()
        else:
            raise ValueError(f'Evaluating metrics by {plot_by} is not supported')

    destination_dir = os.path.dirname(plot_filepath)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    hvplot.save(p, plot_filepath)


if __name__ == '__main__':
    plot_metric(snakemake.output.plot_filepath,
                snakemake.input.interpolated_predictions_filepath,
                snakemake.wildcards['metric'],
                snakemake.wildcards['plot_by'],
                lake_metadata_filepath=snakemake.input.lake_metadata_filepath,
                train_predictions_filepath=snakemake.input.train_predictions_filepath,
                include_train_mean=snakemake.params.include_train_mean,
                doy_bin_width=snakemake.params.doy_bin_width)


