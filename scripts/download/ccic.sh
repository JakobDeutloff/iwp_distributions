#!/bin/bash
# Download all .zarr directories matching a pattern

S3_PREFIX="s3://chalmerscloudiceclimatology/record/cpcir/2008/"
LOCAL_DIR="/work/bm1183/m301049/ccic"

mkdir -p "$LOCAL_DIR"

# List all .zarr directories matching your pattern
aws s3 ls --no-sign-request "$S3_PREFIX" | awk '{print $2}' | grep '^ccic_cpcir_20080101.*\.zarr/$' | while read zarr_dir; do
    # Remove trailing slash
    zarr_dir=${zarr_dir%/}
    echo "Downloading $zarr_dir ..."
    aws s3 cp --no-sign-request --recursive "${S3_PREFIX}${zarr_dir}" "$LOCAL_DIR/$zarr_dir" > /dev/null 2>&1
done