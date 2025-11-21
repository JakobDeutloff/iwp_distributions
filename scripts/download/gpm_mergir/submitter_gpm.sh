#!/bin/bash
#SBATCH --job-name=download_gpm # Specify job name
#SBATCH --output=download_gpm.o%j # name for standard output log file
#SBATCH --error=download_gpm.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# calc iwp
/home/m/m301049/.conda/envs/main/bin/python scripts/download/gpm_mergir/download_gpm.py $1 