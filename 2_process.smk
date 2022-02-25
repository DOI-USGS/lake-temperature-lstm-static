import os
import numpy as np
import pandas as pd

configfile: "2_process/process_config.yaml"


def mntoha_lake_sequence_files(wildcards):
    """
    List all MNTOHA lake sequence .npy files for training and testing.

    :param wildcards: Snakemake wildcards (empty).
    :returns: List of lake training/testing sequence files.

    """
    # make this function dependent on fetch_all
    # needed because lake_metadata.csv is used to determine lake_sequence_files
    lake_metadata_file = checkpoints.fetch_mntoha_metadata.get(**wildcards).output[0]
    lake_metadata = pd.read_csv(lake_metadata_file)
    lake_sequence_files = [os.path.join('2_process/out/mntoha_sequences', f'sequences_{site_id}.npy') for site_id in lake_metadata.site_id]
    return lake_sequence_files


def save_sequences_summary(lake_sequence_files_input, summary_file):
    """
    Summarize lake sequence files
    """
    sequence_counts = []
    for sequences_file in lake_sequence_files_input:
        num_sequences = np.load(sequences_file).shape[0] 
        sequence_counts.append(num_sequences)
    df_counts = pd.DataFrame(data={
        'sequences_file': lake_sequence_files_input,
        'num_sequences': sequence_counts
    })
    df_counts.to_csv(summary_file, index=False)


# Summarize training sequences
rule process_mntoha:
    input:
        mntoha_lake_sequence_files
    output:
        "2_process/out/mntoha_sequences/mntoha_sequences_summary.csv"
    run:
        save_sequences_summary(input, output[0])


# Create .npy of input/output sequences for one lake to use for training and testing
rule mntoha_lake_sequences:
    input:
        config["process_mntoha"]["metadata_augmented_file"],
        config["process_mntoha"]["obs_interpolated_file"],
        config["process_mntoha"]["unzip_log_file"]
    output:
        "2_process/out/mntoha_sequences/sequences_{site_id}.npy"
    params:
        # site_id = lambda wildcards: wildcards.site_id,
        config = config["process_mntoha"]
    script:
        "2_process/src/lake_sequences_mntoha.py"


# Add elevation to MNTOHA lake metadata
rule augment_mntoha_lake_metadata:
    input:
        "1_fetch/out/lake_metadata.csv"
    output:
        "2_process/tmp/mntoha/lake_metadata_augmented.csv"
    script:
        "2_process/src/make_lake_metadata_augmented.py"


def mntoha_obs_file(wildcards):
    '''
    Return MNTOHA temperature observations filepath.

    Depend on unzip_archive checkpoint to ensure that
    temperature_observations.csv gets unzipped.

    :param wildcards: Snakemake wildcards (empty).
    :returns: Path of MNTOHA temperature observations csv

    '''
    # Trigger checkpoint to unzip temperature_observations.csv
    obs_file_directory = checkpoints.unzip_archive.get(
        file_category='obs_mntoha',
        directory_name='temperature_observations'
    ).output[0]
    return os.path.join(obs_file_directory, "temperature_observations.csv")


# Add column of observation depths interpolated to nearest modeling mesh node
rule interpolate_mntoha_obs_depths:
    input:
        config["process_mntoha"]["unzip_log_file"],
    output:
        config["process_mntoha"]["obs_interpolated_file"]
    params:
        depths=config["process_mntoha"]["depths"]
    script:
        "2_process/src/make_obs_interpolated.py"

