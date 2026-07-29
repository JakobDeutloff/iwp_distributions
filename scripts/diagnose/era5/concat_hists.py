# %% 
import xarray as xr 
import pandas as pd
import glob
import re

# %% 
files = glob.glob("/work/bm1183/m301049/era5/diagnosed/iwp_hist*.nc")
files_weighted = [f for f in files if re.search(r"iwp_hist_all_\d{4}\_weighted.nc$", f)]
ds = xr.open_mfdataset(files_weighted).load()

# %%
ds_monthly = ds.resample(time="1M").sum()
ds_monthly["time"] = pd.to_datetime(ds_monthly["time"].dt.strftime("%Y-%m"))

# %%
ds_monthly_interp = ds_monthly.coarsen(iwp=4, boundary="trim").sum()

# %%
ds_monthly_interp.to_netcdf("/work/bm1183/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_all_weighted.nc")
# %% testplot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

mean_hist = ds_monthly_interp['hist'].sum(['time', 'local_time']) / ds_monthly_interp['size'].sum(['time']).sum() 
ax.plot(mean_hist['iwp'], mean_hist, label='Mean Histogram', color='blue')
ax.set_xscale('log')

# %%
