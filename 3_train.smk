
configfile: "3_train/train_config.yaml"

# Train model
rule train_model:
    input:
        npz_filepath = "2_process/out/{data_source}/train.npz"
    output:
        weights_filepath = "3_train/out/{data_source}/{run}/lstm.pt",
        metadata_filepath = "3_train/out/{data_source}/{run}/metadata.npz"
    params:
        config = config
    script:
        "3_train/src/train.py"
