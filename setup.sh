conda activate ltls
module load analytics cuda11.3/toolkit/11.3.0 
export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH
# salloc -N 1 -n 1 -c 1 -p cpu -A watertemp -t 1-23:59:59
