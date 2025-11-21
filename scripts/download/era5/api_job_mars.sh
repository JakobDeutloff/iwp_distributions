#!/bin/bash
#SBATCH --job-name=era5_mars # Specify job name
#SBATCH --output=era5_mars.o%j # name for standard output log file
#SBATCH --error=era5_mars.e%j # name for standard error output log
#SBATCH --partition=shared
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=10GB
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/download/api_retrieval_mars.py $1 $2
