#!/bin/bash
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --partition=choi           # Partition Name (Required)
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=inqlee@ku.edu      # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=32g                     # Job memory request
#SBATCH --time=10-00:00:00             # Time limit days-hrs:min:sec
#SBATCH --output=/home/i243l699/work/src/airway_segmentation/train/monits/monit_UNet_ncase_64_20210913.out
#SBATCH --gres=gpu --constraint=q8000            # 1 GPU
pwd; hostname; date
module load anaconda
conda activate pytorch 
echo "Running python script"
python /home/i243l699/work/src/airway_segmentation/train/main.py
date
