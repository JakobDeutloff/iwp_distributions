# %% 
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.helper_functions import shift_longitudes

# %%
years = np.arange(2006, 2020)
batch_idxs = np.arange(0, 10)
# %% open dataset
def calc_histogram(year, batch_idx):
    ds = xr.open_dataset(f'/work/bm1183/m301049/dardarv3.10/{year}/iwp_dardar_{year}_{batch_idx}.nc').load().pipe(shift_longitudes)
    # get day timestamps
    days = np.unique(ds['time'].dt.floor('D'))
    days = [str(pd.to_datetime(day).date()) for day in days]
    # select tropics and daytime
    local_time = np.array((ds["longitude"].values / 15 + ds["time"].dt.hour.values) % 24)
    mask_daytime = (local_time >= 6) & (local_time <= 18)
    ds_trop = ds.where((ds['latitude'] < 30) & (ds['latitude'] > -30) & mask_daytime, drop=True)
    # calculate histogram
    hist_list = []
    size_list = []
    time_stamps = []
    for day in days:
        bins = bins = np.logspace(-3, 2, 254)[::4]
        len_data = np.isfinite(ds_trop['iwp'].sel(time=day)).sum()
        hist, _ = np.histogram(ds_trop['iwp'].sel(time=day).values, bins=bins, density=False)
        hist_list.append(hist)
        size_list.append(len_data)
        time_stamps.append(day)
    # 
    hists = xr.Dataset(
        {
            "hist": (("bin_center", "time"), np.array(hist_list).T),
            "size": (("time"), np.array(size_list)),
        },
        coords={
            "bin_center": (bins[1:] + bins[:-1]) / 2,
            "time": pd.to_datetime(time_stamps),
        },
    )
    return hists
# %%
hists_list = []
year_batch_pairs = [(year, batch_idx) for year in years for batch_idx in batch_idxs]
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = {executor.submit(calc_histogram, year, batch_idx): (year, batch_idx) for year, batch_idx in year_batch_pairs}
    for future in tqdm(as_completed(futures), total=len(futures)):
        hists = future.result()
        hists_list.append(hists)


# %%
histogram = xr.concat(hists_list, dim='time')

# %% sum up duplicated days
histogram = histogram.groupby('time').sum()

#%% transpose to have bin_center as first dimension
histogram['hist'] = histogram['hist'].transpose('bin_center', 'time')

# %% save
histogram.to_netcdf('/work/bm1183/m301049/dardarv3.10/hist_dardar.nc')
# %%
