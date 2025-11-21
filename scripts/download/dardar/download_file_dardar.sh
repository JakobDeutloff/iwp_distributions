#!/bin/bash
#SBATCH --job-name=iwp_dist# Specify job name
#SBATCH --output=iwp_dist.o%j # name for standard output log file
#SBATCH --error=iwp_dist.e%j # name for standard error output log
#SBATCH --partition=shared
#SBATCH --account=bm1183
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
set -o errexit -o nounset -o pipefail -o xtrace

# Set pythonpath
export PYTHONPATH="/home/m/m301049/iwp_distributions/"

# get arguments
YEAR=$1
DAY=$2


# Configuration
SFTP_HOST="sftp.icare.univ-lille.fr"
SFTP_USER="jakobdeutloff"      # Replace with your ICARE SFTP username
SFTP_PASS="rYg@sihUMT2fd7K"    # Replace with your ICARE SFTP password
LOCAL_DIR="/work/bm1183/m301049/dardarv3.10/$YEAR/$DAY/"               # Local download directory
REMOTE_DIR="/SPACEBORNE/CLOUDSAT/DARDAR-CLOUD.v3.10/$YEAR/$DAY/"       # Folder argument passed to script


# Load lftp module
module load lftp

# Run lftp to mirror the remote folder
lftp -u "$SFTP_USER,$SFTP_PASS" sftp://$SFTP_HOST <<EOF
set ssl:verify-certificate no
mirror --verbose --continue --dereference --parallel=16 "$REMOTE_DIR" "$LOCAL_DIR"
bye
EOF