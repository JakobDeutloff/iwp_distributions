# %%
import xarray as xr
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import time

# %%
year = sys.argv[1]
path = f"/work/bm1183/m301049/dardarv3.10/{year}"

# %% scan all subdirectories under path for .nc files bigger than 400 MB and apply select funtion to them

def is_netcdf_corrupt(file_path):
    try:
        xr.open_dataset(file_path)
        return False  # Not corrupt
    except Exception:
        return True   # Corrupt

def submit_and_wait(script, year, day):
    # Submit job and get job ID
    result = subprocess.run(['sbatch', script, year, day], capture_output=True, text=True)
    job_id = None
    for part in result.stdout.split():
        if part.isdigit():
            job_id = part
            break
    if job_id is None:
        raise RuntimeError("Could not get job ID from sbatch output.")

    # Wait for job to finish
    while True:
        squeue = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        if job_id not in squeue.stdout:
            break
        # wait for 1 min before checking again
        time.sleep(5)  

def select_dardar(file):
    ds = xr.open_dataset(file)
    ds = ds[
        [
            "iwc",
            "ln_iwc_error",
            "latitude",
            "longitude",
            "land_water_mask",
            "day_night_flag",
            "sea_surface_temperature",
        ]
    ]
    os.remove(file)
    ds.to_netcdf(file, mode="w")

def repair_dardar(day, year):
    submit_and_wait("/home/m/m301049/iwp_distributions/scripts/download/download_file_dardar.sh" , str(year), str(day))

# %%
file_day_pairs = []
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        day = root.split("/")[-1]
        file_day_pairs.append((file_path, day))

days = []
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(is_netcdf_corrupt, file_path): day for file_path, day in file_day_pairs}
    for future in tqdm(as_completed(futures)):
        if future.result():
            days.append(futures[future])

# get unique days
days = list(set(days)) 

# %% 
for day in tqdm(days):
    print(day)
    repair_dardar(day, year)
    select_dardar(os.path.join(path, day))

# %%
