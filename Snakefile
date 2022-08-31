include: "1_fetch.smk"
include: "2_process.smk"
include: "3_train.smk"
include: "4_evaluate.smk"


rule all:
    input:
        pull_date = "1_fetch/in/pull_date.txt",
        summary_filepath = "3_train/out/model_prep/summary.csv"

