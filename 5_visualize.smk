include: "4_evaluate.smk"

# Plot a metric (e.g. RMS error) over a dataset (e.g. validation) by a variable (e.g. day of year)
# Example call with output: snakemake -c1 5_visualize/out/model_prep/initial/local_a/rmse-by-doy-over-valid.png
rule plot_metric:
    input:
        interpolated_predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_{dataset}.csv",
        lake_metadata_filepath = "2_process/tmp/{data_source}/lake_metadata_augmented.csv",
        train_predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_train.csv",
    params:
        include_train_mean = True,
        doy_bin_width = 5
    output:
        plot_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/{metric}-by-{plot_by}-over-{dataset}.png"
    script:
        "5_visualize/src/plot_metrics.py"


# Plot all metrics for one model and one dataset
rule plot_all_metrics:
    input:
        interpolated_predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_{dataset}.csv",
        rmse_lake_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-lake-over-{dataset}.png",
        bias_lake_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-lake-over-{dataset}.png",
        rmse_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-depth-over-{dataset}.png",
        bias_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-depth-over-{dataset}.png",
        rmse_doy_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-doy-over-{dataset}.png",
        bias_doy_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-doy-over-{dataset}.png"
    output:
        dummyfile = "5_visualize/out/{data_source}/{run_id}/{model_id}/plot_all_{dataset}.dummy"
    shell:
        "touch {output.dummyfile}"

