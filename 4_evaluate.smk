# Make predictions
rule predict:
    input:
        # locations of model weights and metadata depend on data_source, run_id, and model_id
        weights_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_weights.pt",
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz",
        # path to train, valid, or test.npz depends on data_source but not on run_id or model_id
        dataset_filepath = "2_process/out/{data_source}/{dataset}.npz",
    output:
        predictions_filepath = "4_evaluate/out/{data_source}/{run_id}/{model_id}/preds_{dataset}.npz"
    script:
        "4_evaluate/src/predict.py"

