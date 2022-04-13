include: "1_fetch.smk"
include: "2_process.smk"
include: "3_train.smk"


rule all:
    input:
        pull_date = "1_fetch/in/pull_date.txt",
        weights_filepath = "3_train/out/mntoha/0/lstm.pt"

