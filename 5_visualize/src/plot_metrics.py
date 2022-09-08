import os
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import hvplot.pandas
import holoviews as hv


def load_predictions(filepath, doy_bin_width):
    """
    Load predictions for one dataset from a csv as a pandas Dataframe.
    Load the predictions for the training, validation, or testing set. Remove
    any NaN predictions. Compute residuals (as observed - predicted) and save
    them as an extra column in the dataframe.
    """
    raw_preds = pd.read_csv(filepath, parse_dates=['date'])
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
    Get RMS for a series
    Useful as argument to groupby().aggregate()
    """
    return np.sqrt((df**2).mean())


def plot_by_lake(agg_metric, lake_metadata_filepath, metric):
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


def plot_metric(plot_filepath, predictions_filepath, metric, by,
                lake_metadata_filepath=None,
                train_predictions_filepath=None,
                include_train_mean=False,
                doy_bin_width=1):
    """
    Plot a metric (e.g. RMS error) as a function of some variable (e.g. lake depth)
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
        preds.groupby(prediction_columns[by])['residual']
        .aggregate(agg_functions[metric])
        .rename(f'LSTM_{metric}')
    )

    if by == 'lake':
        # Plotting by lake creates a geographic plot, so call separate function
        p = plot_by_lake(agg_metric, lake_metadata_filepath, metric)
    else:
        if include_train_mean:
            # For comparison, take the mean of the training set and use it as a predictor to get residuals
            # This shows visually where prediction is more challenging due to observation variance
            # or difference from the training data
            train_preds = load_predictions(train_predictions_filepath, doy_bin_width)
            train_mean_temp = train_preds.groupby(prediction_columns[by])['temp'].mean().rename('train_mean_temp')
            preds_train_mean = preds.merge(train_mean_temp, how='left', on=prediction_columns[by]).copy()
            preds_train_mean['train_mean_residual'] = preds_train_mean['temp'] - preds_train_mean['train_mean_temp']
            metric_train_mean = (
                preds_train_mean.groupby(prediction_columns[by])['train_mean_residual']
                .aggregate(agg_functions[metric])
                .rename(f'train_mean_{metric}')
            )
            df_plot = pd.merge(agg_metric, metric_train_mean, how='outer', left_index=True, right_index=True)
        else:
            df_plot = agg_metric

        zero_line = hv.HLine(0).opts(color='black', level='underlay')
        if by == 'depth':
            if metric == 'bias':
                p = df_plot.hvplot.line() * zero_line
            else:
                p = df_plot.hvplot.line()
        elif by == 'doy':
            if metric == 'bias':
                p = df_plot.hvplot.line() * zero_line
            else:
                p = df_plot.hvplot.line()
        else:
            raise ValueError(f'Evaluating metrics by {by} is not supported')

    destination_dir = os.path.dirname(plot_filepath)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    hvplot.save(p, plot_filepath)


if __name__ == '__main__':
    plot_metric(snakemake.output.plot_filepath,
                snakemake.input.interpolated_predictions_filepath,
                snakemake.wildcards['metric'],
                snakemake.wildcards['by'],
                lake_metadata_filepath=snakemake.input.lake_metadata_filepath,
                train_predictions_filepath=snakemake.input.train_predictions_filepath,
                include_train_mean=snakemake.params.include_train_mean,
                doy_bin_width=snakemake.params.doy_bin_width)


