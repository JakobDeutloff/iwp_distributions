# %%
import ccic
import s3fs
import xarray as xr
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr
from tqdm import tqdm
import sys

# %%
year = sys.argv[1]
month = sys.argv[2]
s3 = s3fs.S3FileSystem(anon=True)
prefix = f"chalmerscloudiceclimatology/record/cpcir/{year}/ccic_cpcir_{year}{month}*"
files = s3.glob(prefix)
# %%
local_dir = f"/work/bm1183/m301049/ccic/{year}/"
os.makedirs(local_dir, exist_ok=True)
bins = np.logspace(-3, 2, 254)


def calculate_iwp_distribution(file_path):
    ds = xr.open_zarr(s3.get_mapper(file_path))
    ds = ds.sel(latitude=slice(30, -30)).load()
    container = xr.Dataset(
        {
            "hist": (("bin_center", "time"), np.zeros((len(bins) - 1, 2))),
            "size": (("time"), np.zeros(2)),
        },
        coords={
            "bin_center": bins[:-1],
            "time": ds.time,
        },
    )
    for i in range(2):
        hist, _ = np.histogram(ds["tiwp"].isel(time=i).values, bins=bins, density=False)
        size = ds["tiwp"].isel(time=i).size
        container["hist"][:, i] = hist
        container["size"][i] = size
    return container


# %% concatenate results and save
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(calculate_iwp_distribution, files)))
all_data = xr.concat(results, dim="time")
all_data["hist"].attrs = {
    "description": "Histogram counts of ice water path (IWP) in each bin",
    "units": "counts",
}
all_data["size"].attrs = {
    "description": "Total number of IWP samples used to compute the histogram for each time",
    "units": "counts",
}
path = os.path.join(local_dir, f"ccic_cpcir_iwp_distribution_{year}{month}.nc")
if os.path.exists(path):
    os.remove(path)
all_data.to_netcdf(path)

# %%
