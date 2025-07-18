#!/bin/bash
# Download all .zarr directories matching a pattern in parallel

S3_PREFIX="s3://chalmerscloudiceclimatology/record/cpcir/2008/"
LOCAL_DIR="/work/bm1183/m301049/ccic"
MAX_JOBS=16  # Number of parallel downloads

job_count=0

aws s3 ls --no-sign-request "$S3_PREFIX" | awk '{print $2}' | grep '^ccic_cpcir_200801.*\.zarr/$' | while read zarr_dir; do
    zarr_dir=${zarr_dir%/}
    echo "Downloading $zarr_dir ..."
    aws s3 cp --no-sign-request --recursive "${S3_PREFIX}${zarr_dir}" "$LOCAL_DIR/$zarr_dir" > /dev/null 2>&1 &
    ((job_count++))
    if (( job_count % MAX_JOBS == 0 )); then
        wait
    fi
done

wait  # Wait for any remaining jobs to finish