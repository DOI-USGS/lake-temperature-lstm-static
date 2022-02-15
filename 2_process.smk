import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(1, './2_process/src')
from process_mntoha import make_lake_metadata_augmented
from process_mntoha import make_obs_interpolated
from process_mntoha import unzip_all

# Include is necessary to call get_mntoha_input_files
# This include is redundant unless this file is referenced directly rather than through the main Snakefile
include: "1_fetch.smk"
configfile: "2_process/process_config.yaml"


def mntoha_lake_sequence_files(wildcards):
    """
    List all MNTOHA lake sequence .npy files for training and testing.

    :param wildcards: Snakemake wildcards (unused).
    :returns: List of lake training/testing sequence files.

    """
    # make this function dependent on fetch_all
    # needed because lake_metadata.csv is used to determine lake_sequence_files
    ck_output = checkpoints.fetch_all.get(**wildcards).output[0]
    print(ck_output)
    out_dir = os.path.join(config["process_mntoha"]["out_dir"], 'mntoha_sequences')
    lake_metadata_file = config["process_mntoha"]["metadata_file"]
    lake_metadata = pd.read_csv(lake_metadata_file)
    lake_sequence_files = [os.path.join(out_dir, f'sequences_{site_id}.npy') for site_id in lake_metadata.site_id]
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
        "2_process/src/process_mntoha_lake.py"


# Add elevation to MNTOHA lake metadata
rule augment_mntoha_lake_metadata:
    input:
        config["process_mntoha"]["metadata_file"]
    output:
        config["process_mntoha"]["metadata_augmented_file"]
    run:
        make_lake_metadata_augmented(input[0], output[0])


# Add column of observation depths interpolated to nearest modeling mesh node
rule interpolate_mntoha_obs_depths:
    input:
        config["process_mntoha"]["obs_file"],
        config["process_mntoha"]["unzip_log_file"]
    output:
        config["process_mntoha"]["obs_interpolated_file"]
    run:
        make_obs_interpolated(input[0], output[0], config["process_mntoha"]["depths"])


# Unzip all MNTOHA data downloaded from ScienceBase
rule unzip_mntoha:
    input: 
        get_mntoha_input_files
    output:
        config["process_mntoha"]["unzip_log_file"]
    run:
        unzip_all(input,
                  output[0],
                  config["process_mntoha"]["source_dir"],
                  config["process_mntoha"]["tmp_dir"])

