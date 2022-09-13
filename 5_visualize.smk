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
        bin_width = None
    output:
        plot_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/{metric,rmse|bias}-by-{plot_by}-over-{dataset}.png"
    script:
        "5_visualize/src/plot_metrics.py"


# Plot histograms of observations
rule plot_obs_count:
    input:
        predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_{dataset}.csv",
        lake_metadata_filepath = "2_process/tmp/{data_source}/lake_metadata_augmented.csv"
    output:
        plot_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-{plot_by}-over-{dataset}.png"
    script:
        "5_visualize/src/plot_obs_counts.py"


# Plot all metrics for one model and one dataset
rule plot_all_metrics:
    input:
        interpolated_predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_{dataset}.csv",
        rmse_lake_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-lake-over-{dataset}.png",
        bias_lake_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-lake-over-{dataset}.png",
        rmse_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-depth-over-{dataset}.png",
        bias_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-depth-over-{dataset}.png",
        obs_count_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-depth-over-{dataset}.png",
        rmse_doy_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-doy-over-{dataset}.png",
        bias_doy_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-doy-over-{dataset}.png",
        obs_count_doy_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-doy-over-{dataset}.png",
        rmse_elevation_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-elevation-over-{dataset}.png",
        bias_elevation_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-elevation-over-{dataset}.png",
        obs_count_elevation_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-elevation-over-{dataset}.png",
        rmse_area_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/rmse-by-area-over-{dataset}.png",
        bias_area_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/bias-by-area-over-{dataset}.png",
        obs_count_area_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-area-over-{dataset}.png",
        obs_count_doy_depth_filepath = "5_visualize/out/{data_source}/{run_id}/{model_id}/obs-count-by-doy_depth-over-{dataset}.png",
    output:
        dummyfile = "5_visualize/out/{data_source}/{run_id}/{model_id}/plot_all_{dataset}.dummy"
    shell:
        "touch {output.dummyfile}"


