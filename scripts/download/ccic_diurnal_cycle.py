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
local_dir = f"/work/bm1183/m301049/ccic_daily_cycle/{year}/"
os.makedirs(local_dir, exist_ok=True)
bins = np.arange(0, 25, 1)


def calculate_daily_cycle_distribution(file_path):
    ds = xr.open_zarr(s3.get_mapper(file_path))
    ds = ds.sel(latitude=slice(30, -30)).load()

    container = xr.Dataset(
        {
            "hist": (("local_time", "time"), np.zeros((24, 2))),
            "size": (("time"), np.zeros(2)),
        },
        coords={
            "local_time": 0.5 * (bins[1:] + bins[:-1]),
            "time": ds.time,
        },
    )
    for i in range(2):
        ds_ts = ds.isel(time=i)
        local_time = (
        ds_ts["time"].dt.hour + (ds_ts["time"].dt.minute / 60) + (ds_ts["longitude"] / 15)
        ) % 24
        ds_ts = ds_ts.assign(
            {
                "local_time": (("latitude", "longitude"), np.tile(local_time.values, (ds_ts["latitude"].size, 1))),
            }
        )
        times_conv = ds_ts['local_time'].where(ds_ts['tiwp']>1)
        hist, _ = np.histogram(times_conv.values, bins=bins, density=False)
        size = np.isfinite(times_conv).sum().item()
        container["hist"][:, i] = hist
        container["size"][i] = size
    return container


# %% concatenate results and save
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(calculate_daily_cycle_distribution, files), total=len(files)))
all_data = xr.concat(results, dim="time")
all_data["hist"].attrs = {
    "description": "Histogram counts of IWP > 1 kg/m2 in each local time bin",
    "units": "counts",
}
all_data["size"].attrs = {
    "description": "Total number of IWP samples > 1 kg/m2 used to compute the histogram for each time",
    "units": "counts",
}
path = os.path.join(local_dir, f"ccic_cpcir_daily_cycle_distribution_{year}{month}.nc")
if os.path.exists(path):
    os.remove(path)
all_data.to_netcdf(path)
