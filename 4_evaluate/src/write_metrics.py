import pandas as pd

import sys
sys.path.insert(1, '4_evaluate/src')
sys.path.insert(1, '5_visualize/src')
from plot_metrics import load_predictions, rms


def compute_metrics(predictions_filepath):
    """
    Compute metrics for a dataset
    """
    preds = load_predictions(predictions_filepath)
    rmse_lake = preds.groupby('site_id')['residual'].aggregate(rms).rename('LSTM_rmse')

    results_dict = {
        'RMSE': [rms(preds['residual'])],
        'Median lake RMSE': [rmse_lake.median()],
        'Mean lake RMSE': [rmse_lake.mean()],
    }
    return pd.DataFrame(results_dict)


def write_metrics(metrics_filepath, predictions_filepath):
    """
    Write metrics to file
    """
    metrics = compute_metrics(predictions_filepath)
    # print(rms(preds['residual']), rmse_lake.median(), rmse_lake.mean())
    metrics.to_csv(metrics_filepath)


if __name__ == '__main__':
    write_metrics(snakemake.output.metrics_filepath,
                  snakemake.input.interpolated_predictions_filepath)

