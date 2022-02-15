include: "1_fetch.smk"
include: "2_process.smk"


rule all:
    input:
        "1_fetch/in/pull_date.txt",
        # get_mntoha_input_files,
        "2_process/out/mntoha_sequences/mntoha_sequences_summary.csv"

