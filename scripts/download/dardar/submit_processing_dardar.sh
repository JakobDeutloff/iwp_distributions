#!/bin/bash
#SBATCH --job-name=process_dardar# Specify job name
#SBATCH --output=process_dardar.o%j # name for standard output log file
#SBATCH --error=process_dardar.e%j # name for standard error output log
#SBATCH --partition=compute
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=0
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# Adjust permissions
chmod -R u+rwx /work/bm1183/m301049/dardarv3.10/$1/

# cut
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/download/cut_dardar.py $1 
# calc iwp
/home/m/m301049/.conda/envs/main/bin/python /home/m/m301049/iwp_distributions/scripts/diagnose/dardar_iwp.py $1 