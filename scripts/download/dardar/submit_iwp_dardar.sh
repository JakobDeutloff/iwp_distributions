#!/bin/bash
#SBATCH --job-name=iwp_dardar# Specify job name
#SBATCH --output=iwp_dardar.o%j # name for standard output log file
#SBATCH --error=iwp_dardar.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# calc iwp
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/diagnose/dardar_iwp.py $1 $2