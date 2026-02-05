#!/bin/bash
#SBATCH --job-name=era5_proc # Specify job name
#SBATCH --output=era5_proc.o%j # name for standard output log file
#SBATCH --error=era5_proc.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=mh1126
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/diagnose/era5/calc_stability_era5.py
