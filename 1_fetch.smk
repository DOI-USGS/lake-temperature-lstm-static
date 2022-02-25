import os

configfile: "1_fetch/fetch_config.yaml"


def get_mntoha_input_files(wildcards):
    '''List all MNTOHA files to download
    Includes metadata, temperature observations, and drivers

    :param wildcards: Snakemake wildcards (unused)
    :returns: list of filepaths to be downloaded

    '''
    out_dir = "1_fetch/out"
    categories = config["fetch_mntoha"].keys()
    input_files = []
    # include files under files key in each category
    for category in categories:
        for filename in config["fetch_mntoha"][category]["files"]:
            input_files.append(os.path.join(out_dir, category + "_mntoha", filename))
    # include driver files: 3 types x 13 suffixes = 39 files
    for driver_type in config["fetch_mntoha"]["drivers"]["driver_types"]:
        for suffix in config["fetch_mntoha"]["drivers"]["suffixes"]:
            input_files.append(os.path.join(out_dir, "drivers_mntoha", f"{driver_type}_{suffix}.zip"))
    return input_files


checkpoint fetch_all:
    input:
        "1_fetch/in/pull_date.txt",
        get_mntoha_input_files
    output:
        "1_fetch/out/downloaded_files.txt"
    run:
        with open(output[0], 'w') as f:
            for input_file in input[1]:
                f.write(input_file + '\n')


# Download lake_metadata.csv
# This is a checkpoint because lake_metadata.csv is needed to determine 
# later outputs; namely, the lake_sequence file names.
checkpoint fetch_mntoha_metadata:
    input:
        "1_fetch/in/pull_date.txt",
    output:
        "1_fetch/out/lake_metadata.csv"
    params:
        sb_id = "5e5c1c1ce4b01d50924f27e7"
    script:
        "1_fetch/src/sb_fetch.py"


# Download MNTOHA temperature observations zip file
rule fetch_mntoha_obs:
    input:
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/tmp/obs_mntoha/temperature_observations.zip"
    params:
        sb_id = "5e5d0b68e4b01d50924f2b32"
    script:
        "1_fetch/src/sb_fetch.py"


# Download MNTOHA meteorological drivers, clarity, and ice flag zip files
rule fetch_mntoha_dynamic:
    input:
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/tmp/dynamic_mntoha/{file}"
    params:
        sb_id = "5e5d0b96e4b01d50924f2b34"
    script:
        "1_fetch/src/sb_fetch.py"

