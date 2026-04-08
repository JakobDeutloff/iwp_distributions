# %%
import xarray as xr
import pandas as pd
from src.helper_functions import interpolate_bins
import numpy as np
# %% open CCIC
path = "/work/bm1183/m301049/ccic/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hist = xr.concat(hist_list, dim="time")
hist = hist.rename({"bin_center": "iwp"})
hist = hist.transpose("time", "iwp")
# %%
hist_monthly = hist.resample(time="1ME").sum()
hist_monthly["time"] = pd.to_datetime(hist_monthly["time"].dt.strftime("%Y-%m"))


# %% interpolate bins to match other hists 
new_bins = np.logspace(-3, 2, 254)[::4]
hist_monthly_interp = interpolate_bins(hist_monthly, new_bins, "iwp")

# %% save the interpolated histogram
hist_monthly_interp.to_netcdf("/work/bm1183/m301049/ccic/hists/ccic_monthly_hist_interpolated.nc")

# %%
