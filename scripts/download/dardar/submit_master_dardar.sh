#!/bin/bash
#SBATCH --job-name=master_dardar# Specify job name
#SBATCH --output=master_dardar.o%j # name for standard output log file
#SBATCH --error=master_dardar.e%j # name for standard error output log
#SBATCH --partition=shared
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=1GB
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# cut
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/download/master_dardar.py