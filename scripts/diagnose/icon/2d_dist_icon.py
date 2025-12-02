# %%
import numpy as np
import xarray as xr
import numpy as np
import sys
from tqdm import tqdm
import os

# %% load data
run = sys.argv[1]
print(f"Processing run: {run}")
followups = {"jed0011": "jed0111", "jed0022": "jed0222", "jed0033": "jed0333"}
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

iwp = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/{run}_iwp.nc"
)
iwp = iwp.rename_vars({'__xarray_dataarray_variable__':'iwp'})


# %% select timeslice from datasets 
time_slices = np.arange(0, iwp.time.size, 24)

hists = xr.DataArray(
    np.zeros((len(time_slices) - 1, 24)),
    dims=['day', "local_hour"],
    coords={
        "day": np.arange(0, len(time_slices) - 1),
        "local_hour": np.arange(0, 24)
    }
)
for i in tqdm(range(0, len(time_slices) - 1)):
    start = time_slices[i]
    end = time_slices[i + 1]
    sample = iwp.isel(time=slice(start, end))
    #  calculate local_time
    sample = sample.assign(
        time_local=lambda d: d.time.dt.hour + (d.time.dt.minute / 60) + (d.clon / 15)
    )
    sample["time_local"] = (
        sample["time_local"]
        .where(sample["time_local"] < 24, sample["time_local"] - 24)
        .where(sample["time_local"] > 0, sample["time_local"] + 24)
    )
    # calculate histogram
    bins = np.arange(0, 25, 1)
    hist, edges = np.histogram(
        sample["time_local"].where(sample["iwp"] > 1),
        bins=bins,
        density=False,
    )
    hists[i, :] = hist

# %% save hists
path = f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/deep_clouds_daily_cycle_exact.nc"
if os.path.exists(path):
    os.remove(path)
hists.to_netcdf(path)

