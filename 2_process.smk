import os
import numpy

# Include is necessary to call get_mntoha_input_files
# This include is redundant unless this file is referenced directly rather than through the main Snakefile
include: "1_fetch.smk"
configfile: "2_process/process_config.yaml"

rule process_dynamic_data_mntoha:
    input:
        # This function is defined in 1_fetch.smk
        get_mntoha_input_files
    output:
        "2_process/out/train_mntoha.npz"
    params:
        config = config["process_mntoha"]
    script:
        "2_process/src/process_mntoha.py"

