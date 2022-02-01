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


rule fetch_all:
    input:
        "1_fetch/in/pull_date.txt",
        get_mntoha_input_files


rule fetch_mntoha_data_file:
    input:
        "1_fetch/in/pull_date.txt"
    output:
        "1_fetch/out/{file_category}_mntoha/{file}"
    params:
        sb_id = lambda wildcards: config["fetch_mntoha"][wildcards.file_category]["sb_id"]
    script:
        "1_fetch/src/sb_fetch.py"

