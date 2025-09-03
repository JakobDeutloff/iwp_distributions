# %% import
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# %% def functions
def read_cloudsat(year):
    """
    Function to read CloudSat for a given year
    """

    path_cloudsat = "/work/bm1183/m301049/cloudsat/"
    cloudsat = xr.open_dataset(
        path_cloudsat + year + "-07-01_" + str(int(year) + 1) + "-07-01_fwp.nc"
    )
    # convert ot pandas
    cloudsat = cloudsat.to_pandas()
    # select tropics
    lat_mask = (cloudsat["lat"] <= 30) & (cloudsat["lat"] >= -30)

    return cloudsat[lat_mask]


# %% read in data
years = [str(i) for i in range(2006, 2019)]
cloudsat_list = [read_cloudsat(year) for year in years]
cloudsat = pd.concat(cloudsat_list)

# %% build xarray
cloudsat_xr = xr.Dataset(
    {
        "iwp": (("time"), cloudsat["ice_water_path"].values * 1e-3),
        "time": (("time"), cloudsat["time"].values),
        "lat": (("time"), cloudsat["lat"].values),
        "lon": (("time"), cloudsat["lon"].values),
    }
)

# %% add local time
cloudsat_xr = cloudsat_xr.assign(
    {
        "local_time": (
            ("time"),
            (cloudsat_xr["lon"].values / 15 + cloudsat_xr["time"].dt.hour.values) % 24,
        )
    }
)
# %% calculate histograms for every month

bins = np.logspace(-3, 2, 254)[::4]
mask_daytime = (cloudsat_xr.sel(time="2007-01")["local_time"] >= 6) & (
    cloudsat_xr.sel(time="2007-01")["local_time"] <= 18
)
len_full = np.isfinite(cloudsat_xr['iwp'].sel(time="2007-01").where(mask_daytime)).sum()

# %%
def calc_histogram(timestamp):  # number of full res samples in a month (2007-01)

    mask_daytime = (cloudsat_xr.sel(time=timestamp)["local_time"] >= 6) & (
        cloudsat_xr.sel(time=timestamp)["local_time"] <= 18
    )
    data = cloudsat_xr.sel(time=timestamp)["iwp"].where(mask_daytime)
    len_data = np.isfinite(data).sum()

    if len_data  < 0.5 * len_full:
        print(f"Warning: less than 50% of data available for {timestamp}, skipping")
        return (np.full(len(bins) - 1, np.nan), np.nan)
    else:
        hist, _ = np.histogram(data, bins=bins, density=False)
        return (hist, len_data)


years = np.arange(2006, 2019)
months = [f"{i:02d}" for i in range(1, 13)]
time_stamps = [f"{year}-{month}" for year in years for month in months]
hist_list = []
size_list = []

with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(tqdm(executor.map(calc_histogram, time_stamps)))

# %%Unpack results
hist_list, size_list = zip(*results)

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

# %% save hists 
path = '/work/bm1183/m301049/cloudsat/dists.nc'
hists.to_netcdf(path)

