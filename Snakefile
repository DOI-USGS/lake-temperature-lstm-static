include: "1_fetch.smk"
include: "2_process.smk"


rule all:
    input:
        "1_fetch/in/pull_date.txt",
        get_mntoha_input_files

