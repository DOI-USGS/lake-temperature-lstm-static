
configfile: "3_train/train_config.yaml"

# Save config files to output directory
rule save_config:
    input:
        process_config = "2_process/process_config.yaml",
        train_config = "3_train/train_config.yaml"
    output:
        process_config = "3_train/out/{data_source}/{run_id}/process_config.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/train_config.yaml"
    shell:
        """
        mkdir -p 3_train/out/{wildcards.data_source}/{wildcards.run_id}
        cp {input.process_config} {output.process_config}
        cp {input.train_config} {output.train_config}
        """

# Train model
rule train_model:
    input:
        npz_filepath = "2_process/out/{data_source}/train.npz",
        process_config = "3_train/out/{data_source}/{run_id}/process_config.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/train_config.yaml"
    output:
        weights_filepath = "3_train/out/{data_source}/{run_id}/lstm.pt",
        metadata_filepath = "3_train/out/{data_source}/{run_id}/metadata.npz"
    params:
        config = config,
        run_id = lambda wildcards: wildcards.run_id
    script:
        "3_train/src/train.py"

