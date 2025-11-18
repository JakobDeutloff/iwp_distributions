# %% import
import xarray as xr
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.helper_functions import shift_longitudes

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
cloudsat_xr = cloudsat_xr.pipe(shift_longitudes, lon_name='lon')
bins = np.logspace(-3, 2, 254)[::4]

# %%
lon_min_twp = 120
lon_max_twp = 180
mask = xr.open_dataarray('/work/bm1183/m301049/orcestra/sea_land_mask.nc').pipe(shift_longitudes, lon_name='lon')

def calc_histogram(timestamp):


    # check if any data available
    if not np.any(cloudsat_xr.sel(time=timestamp)["iwp"].values):
        return (np.ones(len(bins) - 1)*np.nan, np.nan)
    

    mask_daytime = (cloudsat_xr.sel(time=timestamp)["local_time"] >= 6) & (
        cloudsat_xr.sel(time=timestamp)["local_time"] <= 18
    )
    # mask_geo = (cloudsat_xr.sel(time=timestamp)["lon"] >= lon_min_twp) & (
    #     cloudsat_xr.sel(time=timestamp)["lon"] <= lon_max_twp
    # )
    mask_sea = mask.sel(lon=cloudsat_xr.sel(time=timestamp)["lon"], lat=cloudsat_xr.sel(time=timestamp)["lat"], method='nearest')
    data = cloudsat_xr.sel(time=timestamp)["iwp"].where(mask_daytime & mask_sea)
    len_data = np.isfinite(data).sum()
    hist, _ = np.histogram(data, bins=bins, density=False)
    return (hist, len_data)

# %% 
years = np.arange(2006, 2019)
months = [f"{i:02d}" for i in range(1, 13)]
time_stamps = [f"{year}-{month}" for year in years for month in months]
hist_list = []
size_list = []

# %%
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
path = '/work/bm1183/m301049/cloudsat/dists_sea.nc'
hists.to_netcdf(path)

# %%
