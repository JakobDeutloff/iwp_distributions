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
s3 = s3fs.S3FileSystem(anon=True)
prefix = f"chalmerscloudiceclimatology/record/cpcir/{year}/ccic_cpcir_{year}*"
files = s3.glob(prefix)
mask = xr.open_dataarray('/work/bm1183/m301049/orcestra/sea_land_mask.nc')
mask = mask.sel(lat=slice(-30, 30)).load()

ds = xr.open_zarr(s3.get_mapper(files[0]))
ds = ds.sel(latitude=slice(30, -30)).load()
del(s3)

mask_sea = mask.sel(lon=ds['longitude'], lat=ds['latitude'], method='nearest').drop_vars(['lon', 'lat'])
# %%
local_dir = f"/work/bm1183/m301049/ccic_daily_cycle/{year}/"
os.makedirs(local_dir, exist_ok=True)
bins_lt = np.arange(0, 25, 1)
bins_iwp = np.logspace(-3, 2, 254)

def calc_2d_hist(file_path):
    s3 = s3fs.S3FileSystem(anon=True)
    ds = xr.open_zarr(s3.get_mapper(file_path))
    ds = ds.sel(latitude=slice(30, -30)).load()

    local_time = (
        ds["time"].dt.hour + (ds["time"].dt.minute / 60) + (ds["longitude"] / 15)
    ) % 24
    local_time = local_time.expand_dims({"latitude": ds["latitude"]}).transpose("time", "latitude", "longitude")

    ds = ds.assign(
        {
            "local_time": (
                ("time", "latitude", "longitude"), local_time.values
                ,
            ),
        }
    )

    hist_1, _, _ = np.histogram2d(
    ds['local_time'].isel(time=0).values.flatten(),
    ds['tiwp'].isel(time=0).values.flatten(),
    bins=[bins_lt, bins_iwp],
    density=False,
    )
    size_1 = np.isfinite(ds['tiwp'].isel(time=0)).sum().item()
    time_1 = ds['time'].isel(time=0).values

    hist_2 , _, _ = np.histogram2d(
    ds['local_time'].isel(time=1).values.flatten(),
    ds['tiwp'].isel(time=1).values.flatten(),
    bins=[bins_lt, bins_iwp],
    density=False,
    )
    size_2 = np.isfinite(ds['tiwp'].isel(time=1)).sum().item()
    time_2 = ds['time'].isel(time=1).values
    return hist_1, hist_2, size_1, size_2, time_1, time_2


# %% concatenate results and save
with ThreadPoolExecutor(max_workers=12) as executor:
    results = list(tqdm(executor.map(calc_2d_hist, files), total=len(files)))
hists_1, hists_2, sizes_1, sizes_2, times_1, times_2 = zip(*results)

# %%
hists = xr.Dataset(
    {
        "hist": (("time", "local_time", "iwp"), np.array(hists_1 + hists_2)),
        "size": (("time"), np.array(sizes_1 + sizes_2)),
    },
    coords={
        "local_time": 0.5 * (bins_lt[1:] + bins_lt[:-1]),
        "iwp": 0.5 * (bins_iwp[1:] + bins_iwp[:-1]),
        "time": np.array(times_1 + times_2).flatten(),
    },
    attrs={
        "description": "2D histogram of CCIC IWP vs local time"
    },
).sortby("time")

# %% save dataset
path = os.path.join(local_dir, f"ccic_cpcir_daily_cycle_distribution_2d_{year}.nc")
hists.to_netcdf(path)

# %%
