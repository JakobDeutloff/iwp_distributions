# %%
import xarray as xr
import numpy as np
import glob 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys

# %% 
path = "/work/mh0010/gridsat_b1/"
year = sys.argv[1]
files = glob.glob(f'{path}/{year}/GRIDSAT-B1*.nc')

# %%
def calc_2d_hist(file_path):
    bins_bt = np.arange(175, 330, 1)
    bins_lt = np.arange(0, 24.1, 0.1)

    ds = xr.open_dataset(file_path)
    ds = ds.sel(lat=slice(-30, 30))

    local_time=(
        (ds["time"].dt.hour + (ds["time"].dt.minute / 60) + (ds["lon"] / 15)) % 24
    )
    ds = ds.assign(
                {
                    "local_time": (("time", "lat", "lon"), np.tile(local_time.values, (1, ds["lat"].size, 1))),
                }
            )
    H, _, _ = np.histogram2d(ds['local_time'].values.flatten(), ds["irwin_cdr"].values.flatten(), bins=[bins_lt, bins_bt], density=False)
    size = np.isfinite(ds["irwin_cdr"]).sum().item()
    return H, size, ds['time'].values 



# %%
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(tqdm(executor.map(calc_2d_hist, files), total=len(files)))
hists, sizes, times = zip(*results)

# %%
bins_lt = np.arange(0, 27, 3)
bins_bt = np.arange(175, 330, 1)
hists = xr.Dataset(
    {
        "hist": (("time", "local_time", "bt"), np.array(hists)),
        "size": (("time"), np.array(sizes)),
    },
    coords={
        "local_time": 0.5 * (bins_lt[1:] + bins_lt[:-1]),
        "bt": 0.5 * (bins_bt[1:] + bins_bt[:-1]),
        "time": np.array(times).flatten(),
    },
    attrs={"description": "2D histogram of Gridsat IR 11m brightness temperature vs local time" },
)

# %% save dataset
out_path = '/work/bm1183/m301049/gridsat/coarse'
hists.to_netcdf(f"{out_path}/gridsat_2d_hist{year}.nc")

