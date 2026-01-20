# %%
import xarray as xr
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
import os

# %% configure
path = "/work/bm1183/m301049/GPM_MERGIR/"
year = sys.argv[1]
region = sys.argv[2]

# %% load first timestep of data
files = glob.glob(f"{path}/merg_{year}*.nc4")
ds = xr.open_dataset(files[0], engine='netcdf4').sel(lat=slice(-30,30))

#%% configure mask
if region == "sea":
    mask = xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")
    mask = mask.sel(lat=slice(-30, 30)).load()
    mask = mask.sel(lon=ds.lon, lat=ds.lat, method='nearest')
    mask['lon'] = ds['lon']
    mask['lat'] = ds['lat']
elif region == "all":
    mask = True
else:
    raise ValueError("region must be 'sea' or 'all'")

# %%
bins_bt = np.arange(150, 340, 1)
# insert 0 in first position
bins_bt = np.insert(bins_bt, 0, 0)
bins_lt = np.arange(0, 25, 1)


def calc_2d_hist(file_path):

    ds = xr.open_dataset(file_path, engine='netcdf4')
    ds = ds.sel(lat=slice(-30, 30))

    local_time = (
        ds["time"].dt.hour + (ds["time"].dt.minute / 60) + (ds["lon"] / 15)
    ) % 24
    local_time = local_time.expand_dims({"lat": ds["lat"]}).transpose("time", "lat", "lon")

    ds = ds.assign(
        {
            "local_time": (
                ("time", "lat", "lon"), local_time.values
                ,
            ),
        }
    )
    H_1, _, _ = np.histogram2d(
        ds["local_time"].isel(time=0).where(mask).values.flatten(),
        ds["Tb"].isel(time=0).where(mask).values.flatten(),
        bins=[bins_lt, bins_bt],
        density=False,
    )
    size_1 = np.isfinite(ds["Tb"].isel(time=0).where(mask)).sum().item()
    H_2, _, _ = np.histogram2d(
        ds["local_time"].isel(time=1).where(mask).values.flatten(),
        ds["Tb"].isel(time=1).where(mask).values.flatten(),
        bins=[bins_lt, bins_bt],
        density=False,
    )
    size_2 = np.isfinite(ds["Tb"].isel(time=1).where(mask)).sum().item()
    return H_1, H_2, size_1, size_2, ds['time'].values[0], ds['time'].values[1]

# %%
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(tqdm(executor.map(calc_2d_hist, files), total=len(files)))
hists_1, hists_2, sizes_1, sizes_2, times_1, times_2 = zip(*results)

# %%
hists = xr.Dataset(
    {
        "hist": (("time", "local_time", "bt"), np.array(hists_1 + hists_2)),
        "size": (("time"), np.array(sizes_1 + sizes_2)),
    },
    coords={
        "local_time": 0.5 * (bins_lt[1:] + bins_lt[:-1]),
        "bt": 0.5 * (bins_bt[1:] + bins_bt[:-1]),
        "time": np.array(times_1 + times_2).flatten(),
    },
    attrs={
        "description": "2D histogram of GPM_MERGIR IR 11m brightness temperature vs local time"
    },
).sortby("time")

# %% save dataset
out_path = f"/work/bm1183/m301049/GPM_MERGIR/hists/gpm_2d_hist_{region}_{year}.nc"
if os.path.exists(out_path):
    os.remove(out_path)
hists.to_netcdf(out_path)
