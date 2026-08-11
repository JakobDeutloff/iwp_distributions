# %%
import numpy as np
import xarray as xr
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd

# %% load data
run = 'jed0011'
print(f"Processing run: {run}")
configs = {"jed0011": "icon-mpim", "jed0022": "icon-mpim-4K", "jed0033": "icon-mpim-2K"}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

iwp = xr.open_dataset(
    f"/work/bu1562/m301049/icon_hcap_data/{names[run]}/production/{run}_iwp.nc"
)
ds = iwp.rename_vars({'__xarray_dataarray_variable__':'iwp'})
# %% get weights
grid_area = xr.open_dataset('/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc')["cell_area_p"].load()
grid_area = grid_area.rename({'cell': 'ncells'})        
grid_area = grid_area.assign_coords(ncells=np.arange(grid_area.sizes['ncells'])).sel(ncells=ds.ncells)
weights_vals = grid_area / grid_area.mean()
#  build weights that has same shape as ds on a day 
weights = weights_vals.expand_dims({"time": ds["time"]}).transpose("time", "ncells")

# %% define bins 
bins_lt = np.arange(0, 25, 1)
bins_iwp = np.logspace(-3, 2, 254)

# %% select timeslice from datasets 
days = np.unique(ds.time.dt.floor('D').values).astype(str)
days = [day.split('T')[0] for day in days]
def calc_2d_hist(day):

    ds_sel = ds.sel(time=day).load()
    weights_sel = weights.sel(time=day).load()

    local_time = (
    ds_sel["time"].dt.hour + (ds_sel["time"].dt.minute / 60) + (ds_sel["clon"] / 15)
    ) % 24
    ds_sel = ds_sel.assign(
    {
        "local_time": (
            ('time', 'ncells'), local_time.data
            ,
        ),
    })
    hist, _, _ = np.histogram2d(
    ds_sel["local_time"].values.flatten(),
    ds_sel["iwp"].values.flatten(),
    bins=[bins_lt, bins_iwp],
    weights=weights_sel.values.flatten(),
    density=False,
    )
    size = weights_sel.where(np.isfinite(ds_sel["iwp"])).sum().item()
    return hist, size

# %%
with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(tqdm(executor.map(calc_2d_hist, days), total=len(days)))

hists, sizes = zip(*results)

# %% construct dataset
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
        "description": "2D histogram of ICON IWP vs local time"
    },
).sortby("time")

# %% save hists
path = f"/work/bu1562/m301049/icon_hcap_data/{names[run]}/production/daily_cycle_hist_weighted.nc"
if os.path.exists(path):
    os.remove(path)
hists_xr.to_netcdf(path)

# %%
