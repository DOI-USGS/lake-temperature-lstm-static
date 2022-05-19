#!/bin/bash
#
run_id=skateboard
model_id=local_a
log_dir=log/${run_id}_${model_id}
mkdir -p ${log_dir}
snakemake --printshellcmds --keep-going --cores all --jobs 1 --rerun-incomplete 3_train/out/mntoha/${run_id}/${model_id}_weights.pt 2>&1 | tee ${log_dir}/run.out
