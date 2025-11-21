# %%
import xarray as xr
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# %%
year = sys.argv[1]
path = f"/work/bm1183/m301049/dardarv3.10/{year}"

# %% scan all subdirectories under path for .nc files bigger than 400 MB and apply select funtion to them

def select_dardar(file):
    try:
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
    except Exception:
        print(f"{file} is corrupt, deleting.")
        os.remove(file)
        return


with ProcessPoolExecutor(max_workers=64) as executor:
    futures = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".nc"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 100 * 1024 * 1024:
                    futures.append(executor.submit(select_dardar, file_path))
    for future in tqdm(futures):
        future.result()

# %%
