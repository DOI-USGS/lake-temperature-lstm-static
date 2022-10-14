import pandas as pd

import sys
sys.path.insert(1, '4_evaluate/src')
sys.path.insert(1, '5_visualize/src')
from plot_metrics import load_predictions, rms

"""
Write evaluation metrics for one model to a csv file. This will be a single row in a csv.
"""


def compute_metrics(predictions_filepath):
    """
    Compute metrics for a dataset, including overall RMSE, median and mean RMSE
    by lake, the total number of observations, and the number of lakes.

    :param predictions_filepath: Path to csv with model predictions
    :returns: DataFrame of metrics computed for the predictions

    """
    preds = load_predictions(predictions_filepath)
    rmse_lake = preds.groupby('site_id')['residual'].aggregate(rms)

    results_dict = {
        'RMSE': [rms(preds['residual'])],
        'Median lake RMSE': [rmse_lake.median()],
        'Mean lake RMSE': [rmse_lake.mean()],
        'Number observations': preds.shape[0],
        'Number lakes': rmse_lake.shape[0]
    }
    return pd.DataFrame(results_dict)


def write_metrics(metrics_filepath, predictions_filepath):
    """
    Write out model evaluation metrics to csv file.

    :param metrics_filepath: Path to output csv for metrics
    :param predictions_filepath: Path to csv with model predictions

    """
    metrics = compute_metrics(predictions_filepath)
    metrics.to_csv(metrics_filepath)


if __name__ == '__main__':
    write_metrics(snakemake.output.metrics_filepath,
                  snakemake.input.interpolated_predictions_filepath)

