#!/bin/bash
#SBATCH --job-name=era5_latlon # Specify job name
#SBATCH --output=era5_latlon.o%j # name for standard output log file
#SBATCH --error=era5_latlon.e%j # name for standard error output log
#SBATCH --partition=interactive
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=10G
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# execute python script in respective environment 
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/download/era5/api_retrieval_copernicus.py
