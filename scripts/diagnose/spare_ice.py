# %%
import xarray as xr 
import os
import pandas as pd
import numpy as np
from  concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# %%
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %%
def open_spareice(path):
    files = []
    for file in os.listdir(path):
        if file.endswith("allsky.nc"):
            files.append(file)
    files.sort()
    ds = xr.open_mfdataset(f"{path}*allsky.nc",
        combine="nested",
        concat_dim="time",
    )
    # get timestamps of files and add as time coordinate to ds
    timestamps = [file.split("_")[2] for file in files]
    ds = ds.assign_coords(time=("time", pd.to_datetime(timestamps)))
    return ds.load()

def calc_histogram(ds):
    bins = np.logspace(-3, 2, 254)[::4]
    centers = 0.5 * (bins[1:] + bins[:-1])
    # get histogram in tropics
    iwp_hist = ds['IWP_lat_hist'].sel(lat_center=slice(-30, 30)).sum(['lat_center'])
    iwp_hist['IWP_bincenters'] = iwp_hist['IWP_bincenters'] * 1e-3
    sizes = ds['N_pixels_lat'].sel(lat_center=slice(-30, 30)).sum(['lat_center'])
    # interpolate to new bins 
    iwp_hist = iwp_hist.interp(IWP_bincenters=centers, method='linear')

    hist = xr.Dataset(
        {
            "hist": (("bin_center", "time"), iwp_hist.values.T),
            "sizes": (("time"), sizes.values),
        },
        coords={"bin_center": centers, "time": ds.time.values},
    )

    # create monthly values
    hist = hist.resample(time="1ME").sum()
    hist['time'] = pd.to_datetime(hist['time'].dt.strftime('%Y-%m'))

    return hist

def process_year(year):
    path = f"{base_path}{satellites[year]}/{year}/"
    ds = open_spareice(path)
    hist = calc_histogram(ds)
    return hist
# %%
base_path = "/work/um0878/users/mbrath/SPARE-ICE-Project/results/gridded_data/"

years = [str(i) for i in range(2007, 2026)]
satellites = {year: "metopb" if int(year) >= 2021 else "metopa" for year in years}

data = [process_year(year) for year in tqdm(years)]

# %% concatenate data and save 
data_all = xr.concat(data, dim='time')

# %%
data_all.to_netcdf('/work/bm1183/m301049/spareice/hists_metop.nc')
# %%
