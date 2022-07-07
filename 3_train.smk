import yaml

# Save config files to output folder
rule save_config:
    input:
        process_config = "2_process/process_config.yaml",
        # location of training config depends on data_source and run_id
        train_config = "3_train/in/{data_source}/{run_id}.yaml"
    output:
        process_config = "3_train/out/{data_source}/{run_id}/{run_id}_process.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/{run_id}_train.yaml"
    shell:
        """
        mkdir -p 3_train/out/{wildcards.data_source}/{wildcards.run_id}
        cp {input.process_config} {output.process_config}
        cp {input.train_config} {output.train_config}
        """


def read_train_config(data_source, run_id):
    """
    Read a training configuration yaml file and return a dictionary

    :param data_source: Source of training data, such as "mntoha".
    :param run_id: ID of the current training run (experiment).

    :returns: Training config dictionary

    """
    config_path = f"3_train/in/{data_source}/{run_id}.yaml"
    with open (config_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


# Train a model and save weights and metadata
# Example:
# $ snakemake --cores all 3_train/out/mntoha/4/hidden_size-50_weights.pt
rule train_model:
    input:
        train_npz_filepath = "2_process/out/{data_source}/train.npz",
        valid_npz_filepath = "2_process/out/{data_source}/valid.npz",
        # Include configs as inputs in order to trigger save_config rule,
        # and save configs to the output folder
        process_config = "3_train/out/{data_source}/{run_id}/{run_id}_process.yaml",
        train_config = "3_train/out/{data_source}/{run_id}/{run_id}_train.yaml"
    output:
        weights_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_weights.pt",
        metadata_filepath = "3_train/out/{data_source}/{run_id}/{model_id}_metadata.npz"
    params:
        # Read in the run-specific config
        config = lambda wildcards: read_train_config(
            wildcards.data_source,
            wildcards.run_id
        ),
        # Pass the run_id and model_id wildcards as params so that they can be
        # saved into metadata easily
        run_id = lambda wildcards: wildcards.run_id,
        model_id = lambda wildcards: wildcards.model_id
    # Change this number of threads as needed. It acts as a maximum.
    threads:
        80
    script:
        "3_train/src/train.py"

# Summarize all trained models in a directory using their metadata
# The parent directory could be a data_source, e.g., mntoha, or a specific run
# that produced many models, e.g., mntoha/5
rule summarize_trained_models:
    # No inputs because we don't want this rule to trigger the train_model rule
    output: 
        summary_filepath = "3_train/out/{parent_directory}/summary.csv"
    script:
        "3_train/src/summarize.py"

