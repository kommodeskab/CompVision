#!/bin/sh

# SET JOB NAME
#BSUB -J cnn_vs_unet

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
#BSUB -W 24:00
#BSUB -o hpc/%J.out 
#BSUB -e hpc/%J.err

module load python3/3.11.9
source .venv/bin/activate
# ph2 dataset
python main.py --batch_size=4 --experiment=segmentation --network=cnn --dataset=ph2 --loss=bce --img_size=256 --run_name=cnn_ph2
python main.py --batch_size=4 --experiment=segmentation --network=unet --dataset=ph2 --loss=bce --img_size=256 --run_name=unet_ph2

# drive dataset
python main.py --batch_size=4 --experiment=segmentation --network=cnn --dataset=drive --loss=bce --img_size=256 --run_name=cnn_drive
python main.py --batch_size=4 --experiment=segmentation --network=unet --dataset=drive --loss=bce --img_size=256 --run_name=unet_drive

