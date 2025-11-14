#!/bin/sh

# SET JOB NAME
#BSUB -J clicks

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
#BSUB -W 4:00
#BSUB -o hpc/%J.out 
#BSUB -e hpc/%J.err

module load python3/3.11.9
source .venv/bin/activate

# python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=1_clicks --num_pos_clicks=1 --num_neg_clicks=1 --max_steps=5000
# python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=2_clicks --num_pos_clicks=2 --num_neg_clicks=2 --max_steps=5000
# python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=5_clicks --num_pos_clicks=5 --num_neg_clicks=5 --max_steps=5000
python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=15_clicks --num_pos_clicks=15 --num_neg_clicks=15 --max_steps=5000
python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=25_clicks --num_pos_clicks=25 --num_neg_clicks=25 --max_steps=5000
# python main.py --batch_size=4 --experiment=clicks --network=unet --dataset=ph2 --loss=point_supervision --img_size=256 --run_name=50_clicks --num_pos_clicks=50 --num_neg_clicks=50 --max_steps=5000