#!/bin/sh

# SET JOB NAME
#BSUB -J losses

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

python main.py --batch_size=4 --experiment=losses --network=unet --dataset=ph2 --loss=bce --img_size=256 --run_name=bce
python main.py --batch_size=4 --experiment=losses --network=unet --dataset=ph2 --loss=weighted_bce --img_size=256 --run_name=weighted_bce
python main.py --batch_size=4 --experiment=losses --network=unet --dataset=ph2 --loss=focal --img_size=256 --run_name=focal