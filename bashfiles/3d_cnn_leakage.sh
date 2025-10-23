#!/bin/sh

# SET JOB NAME
#BSUB -J 3d_cnn_leakage

# select gpu, choose gpuv100, gpua100 or p1 (h100)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=1GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 5:59
#BSUB -o hpc/%J.out 
#BSUB -e hpc/%J.err

module load python3/3.11.9
source .venv/bin/activate
python3 main.py --leakage --batch_size=32 --optimizer=adamw --experiment=3d_cnn --num_workers=12