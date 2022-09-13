import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(1, '5_visualize/src')
from plot_metrics import load_predictions

def plot_obs_count(predictions_filepath, lake_metadata_filepath, plot_by, **kwargs):
    """
    Plot the density of observation data as a histogram or hexbin plot, as a
    function of some variable (e.g. lake depth).

    :param predictions_filepath: Path to predictions csv to load
    :param lake_metadata_filepath: Path to lake metadata csv
    :param plot_by: Independent variable for plot: 
        'elevation', 'area', 'depth', 'doy', or 'doy_depth'. If 'doy_depth', a
        hexbin is plotted with day of year on the horizontal axis and depth on
        the vertical axis.
    :param **kwargs: Keyword arguments to pass through to call to library
        plotting function (sns.histplot or plt.hexbin)
    :returns: Matplotlib figure

    """

    preds = load_predictions(predictions_filepath, lake_metadata_filepath=lake_metadata_filepath)

    fig, ax = plt.subplots()
    if plot_by == 'area':
        # Number of observations by lake surface area
        min_value = preds['area'].min()
        max_value = preds['area'].max()
        logbins = np.logspace(np.log10(min_value), np.log10(max_value), 50)
        hist = sns.histplot(preds['area'], bins=logbins, ax=ax, **kwargs)
        plt.xscale('log')
        ax.set_xlabel('Lake surface area')
        ax.set_ylabel('Number of observations')
    elif plot_by == 'depth':
        # Number of observations by depth
        hist = sns.histplot(preds['interpolated_depth'], bins=50, ax=ax, **kwargs)
        # plt.yscale('log')
        ax.set_xlabel('Lake depth')
        ax.set_ylabel('Number of observations')
    elif plot_by == 'elevation':
        # Number of observations by elevation
        hist = sns.histplot(preds['elevation'], bins=50, ax=ax, **kwargs)
        # plt.yscale('log')
        ax.set_xlabel('Elevation')
        ax.set_ylabel('Number of observations')
    elif plot_by == 'doy':
        # Number of observations by day of year
        hist = sns.histplot(preds['doy'], bins=50, ax=ax, **kwargs)
        ax.set_xlabel('Day of year')
        ax.set_ylabel('Number of observations')
    elif plot_by == 'doy_depth':
        # Number of observations by day of year and depth
        fig, ax = plt.subplots()
        hexbin = ax.hexbin(preds.doy, preds.depth, bins='log', **kwargs)
        cbar = plt.colorbar(hexbin)
        cbar.set_label('Number of observations')
        ax.set_xlabel('Day of year')
        ax.set_ylabel('Lake depth (meters)')
        ax.invert_yaxis()
    else:
        raise ValueError(f'Plotting observation counts by {plot_by} is not supported')
    return fig

def save_obs_count_plot(plot_filepath, predictions_filepath, lake_metadata_filepath, plot_by, **kwargs):
    """
    Save a plot of the density of observation data as a histogram or hexbin
    plot, as a function of some variable (e.g. lake depth) to a file.

    :param plot_filepath: Path to save plot image to. Image format is
        determined from extension.
    :param predictions_filepath: Path to predictions csv to load
    :param lake_metadata_filepath: Path to lake metadata csv
    :param plot_by: Independent variable for evaluating metric: 
        'depth', 'doy', 'elevation', or 'area'
    :param **kwargs: Keyword arguments to pass through to call to library
        plotting function (sns.histplot or plt.hexbin)

    """
    fig = plot_obs_count(predictions_filepath, lake_metadata_filepath, plot_by, **kwargs)
    fig.savefig(plot_filepath)


if __name__ == '__main__':
    save_obs_count_plot(snakemake.output.plot_filepath,
                        snakemake.input.predictions_filepath,
                        snakemake.input.lake_metadata_filepath,
                        snakemake.wildcards['plot_by'])


