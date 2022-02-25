import os


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

