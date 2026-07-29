# %%
import s3fs
import xarray as xr
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xarray as xr
from tqdm import tqdm
import sys
from src.helper_functions import shift_longitudes
import pandas as pd

# %%
year = sys.argv[1]
region = sys.argv[2] 

# %%
ds_full = xr.open_dataset(f"/work/bm1183/m301049/era5/hourly/iwp_{year}.nc").pipe(shift_longitudes)
ds_full['iwp'] = ds_full['tciw'] + ds_full['tcsw']
ds_full = ds_full[['iwp']]

# %% calculate weights
weights_vals = np.cos(np.deg2rad(ds_full.latitude))
# make weights a DataArray with lat and lon and time dims from ds
weights_vals = np.repeat(weights_vals.values[:, np.newaxis], ds_full.longitude.size, axis=1)
weights_vals = np.repeat(weights_vals[np.newaxis, :, :], ds_full.valid_time.size, axis=0)
weights = xr.DataArray(weights_vals, dims=["valid_time", "latitude", "longitude"], coords={"valid_time": ds_full.valid_time, "latitude": ds_full.latitude, "longitude": ds_full.longitude})

# %% configure mask
if region == "sea":
    mask = xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc").pipe(shift_longitudes, lon_name="lon")
    mask = mask.sel(lat=slice(-30, 30)).load()
    mask = mask.sel(
        lon=ds_full["longitude"], lat=ds_full["latitude"], method="nearest"
    ).drop_vars(["lon", "lat"])
elif region == "all":
    mask = True
else:
    raise ValueError("region must be 'sea' or 'all'")
# %%
bins_lt = np.arange(0, 25, 1)
bins_iwp = np.logspace(-3, 2, 254)

days = np.unique(ds_full.valid_time.dt.floor('D').values).astype(str)
days = [day.split('T')[0] for day in days]

def calc_2d_hist(day):

    ds_sel = ds_full[['iwp']].sel(valid_time=day)
    weights_sel = weights.sel(valid_time=day)

    local_time = (
        ds_sel["valid_time"].dt.hour + (ds_sel["valid_time"].dt.minute / 60) + (ds_sel["longitude"] / 15)
    ) % 24
    local_time = local_time.expand_dims({"latitude": ds_sel["latitude"]}).transpose(
        "valid_time", "latitude", "longitude"
    )

    ds_sel = ds_sel.assign(
        {
            "local_time": (
                ("valid_time", "latitude", "longitude"),
                local_time.values,
            ),
        }
    )

    hist, _, _ = np.histogram2d(
    ds_sel["local_time"].where(mask).values.flatten(),
    ds_sel['iwp'].where(mask).values.flatten(),
    bins=[bins_lt, bins_iwp],
    density=False,
    weights=weights_sel.where(mask).values.flatten()
    )
    size = weights_sel.where(np.isfinite(ds_sel["iwp"].where(mask))).sum().item()
    return hist, size


# %%
with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(tqdm(executor.map(calc_2d_hist, days), total=len(days)))
hists, sizes = zip(*results)


# %%
hists_xr = xr.Dataset(
    {
        "hist": (("time", "local_time", "iwp"), np.array(hists)),
        "size": (("time"), np.array(sizes)),
    },
    coords={
        "local_time": 0.5 * (bins_lt[1:] + bins_lt[:-1]),
        "iwp": 0.5 * (bins_iwp[1:] + bins_iwp[:-1]),
        "time": pd.to_datetime(days),
    },
    attrs={
        "description": "2D histogram of ERA5 IWP vs local time"
    },
).sortby("time")

# %% save dataset
path = f"/work/bm1183/m301049/era5/diagnosed/iwp_hist_{region}_{year}_weighted.nc"
hists_xr.to_netcdf(path)


