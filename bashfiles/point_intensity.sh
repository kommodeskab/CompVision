#!/bin/sh

# SET JOB NAME
#BSUB -J intensity

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
#BSUB -W 12:00
#BSUB -o hpc/%J.out 
#BSUB -e hpc/%J.err

module load python3/3.11.9
source .venv/bin/activate
for intensity in 1 10 20 30 40 50
do
    python main.py --batch_size=4 --experiment=intensity --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=${intensity}_intensity --num_pos_clicks=10 --num_neg_clicks=10 --max_steps=10000 --intensity=${intensity}
done