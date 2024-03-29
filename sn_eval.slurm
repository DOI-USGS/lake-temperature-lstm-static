#!/bin/bash
#
#SBATCH --account=watertemp
#SBATCH --partition=cpu
#SBATCH --job-name=snakemake
#SBATCH --time=1-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=log/sbatch_all_%a_%A.out
#SBATCH --error=log/sbatch_all_%a_%A.out

## Initialize work environment
source ~/.bashrc
conda activate ltls
module load analytics cuda11.3/toolkit/11.3.0

## Run the main job

run_id=initial
model_id=cpu_a
data_source=model_prep
log_dir=log/${data_source}_${run_id}_${model_id}
mkdir -p ${log_dir}
# From Jeff's blogpost: https://github.com/jsadler2/wdfn-blog/blob/ca63d4dc4bf6aeb6962a58953057a41c9db1ad8a/content/snakemake-ml-experiments.md
# Also see snakemake profiles: https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles
# snakemake --cluster "sbatch -A watertemp -t 1:59:59 -p cpu -N 1 -n 1 -c 1 --job-name=${data_source}_${run_id}_${model_id} -e ${log_dir}/slurm-%j.out -o ${log_dir}/slurm-%j.out" --printshellcmds --keep-going --cores all --jobs 8 --rerun-incomplete -s 5_visualize.smk 5_visualize/out/${data_source}/${run_id}/${model_id}/plot_all_valid.dummy 2>&1 | tee ${log_dir}/plot.out
snakemake --cluster "sbatch -A watertemp -t 1:59:59 -p cpu -N 1 -n 1 -c 1 --job-name=${data_source}_${run_id}_${model_id} -e ${log_dir}/slurm-%j.out -o ${log_dir}/slurm-%j.out" --printshellcmds --keep-going --cores all --jobs 8 --rerun-incomplete -s 5_visualize.smk 4_evaluate/out/${data_source}/${run_id}/${model_id}/metrics_valid.csv 2>&1 | tee ${log_dir}/eval_valid.out
snakemake --cluster "sbatch -A watertemp -t 1:59:59 -p cpu -N 1 -n 1 -c 1 --job-name=${data_source}_${run_id}_${model_id} -e ${log_dir}/slurm-%j.out -o ${log_dir}/slurm-%j.out" --printshellcmds --keep-going --cores all --jobs 8 --rerun-incomplete -s 5_visualize.smk 4_evaluate/out/${data_source}/${run_id}/${model_id}/metrics_train.csv 2>&1 | tee ${log_dir}/eval_train.out
snakemake --cluster "sbatch -A watertemp -t 1:59:59 -p cpu -N 1 -n 1 -c 1 --job-name=${data_source}_${run_id}_${model_id} -e ${log_dir}/slurm-%j.out -o ${log_dir}/slurm-%j.out" --printshellcmds --keep-going --cores all --jobs 8 --rerun-incomplete -s 5_visualize.smk 4_evaluate/out/${data_source}/${run_id}/${model_id}/metrics_test.csv 2>&1 | tee ${log_dir}/eval_test.out
