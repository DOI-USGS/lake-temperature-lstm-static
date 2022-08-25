# Save unnormalized dataset
rule unnormalize_dataset:
    input: 
        # location of metadata depends on data_source, run_id, and model_id
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz",
        # path to train, valid, or test.npz depends on data_source but not on run_id or model_id
        dataset_filepath = "2_process/out/{data_source}/{dataset}.npz"
    output:
        output_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/unnormalized_{dataset}.npz"
    script:
        "4_evaluate/src/unnormalize_dataset.py"


# Make predictions
rule predict_dataset:
    input:
        # locations of model weights and metadata depend on data_source, run_id, and model_id
        weights_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_weights.pt",
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz",
        # path to train, valid, or test.npz depends on data_source but not on run_id or model_id
        dataset_filepath = "2_process/out/{data_source}/{dataset}.npz",
    output:
        predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/preds_{dataset}.npz"
    script:
        "4_evaluate/src/predict_dataset.py"


# Interpolate predictions to observation locations
rule interpolate_predictions:
    input:
        predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/preds_{dataset}.npz",
        # location of metadata depends on data_source, run_id, and model_id
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz",
        # path to train, valid, or test.npz depends on data_source but not on run_id or model_id
        dataset_filepath = "2_process/out/{data_source}/{dataset}.npz",
        obs_file = "2_process/tmp/{data_source}/temperature_observations_interpolated.csv"
    params:
        batch_size = 100
    output:
        interpolated_predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/interpolated_predictions_{dataset}.csv"
    script:
        "4_evaluate/src/interpolate_predictions.py"


# Plot RMS error over dataset as a function of depth
rule plot_all_metrics:
    input:
        predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/preds_{dataset}.npz"
    output:
        rmse_by_depth_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/plots/rmse-by-depth-{dataset}.png"
    script:
        "4_evaluate/src/plot_metrics.py"

