
def write_metrics(metrics_filepath, predictions_filepath):
    preds = load_predictions(predictions_filepath)
    rmse_lake = preds.groupby('site_id')['residual'].aggregate(rms).rename('LSTM_rmse')
    print(rms(preds['residual']), rmse_lake.median(), rmse_lake.mean())


if __name__ == '__main__':
    write_metrics(snakemake.output.metrics_filepath,
                  snakemake.input.interpolated_predictions_filepath)

