# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.signal import detrend

# %% open distributions 
path = '/work/bm1183/m301049/ccic/'
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc")
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

# %% concatenate hists 
hists = xr.concat(hist_list, dim='time')

# %% coarsen histograms and normalise by size 
hists_coarse = hists.coarsen(bin_center=4, boundary='trim').sum()
hists_norm = hists_coarse['hist'] / hists_coarse['size']
bins = bins = np.logspace(-3, 2, 254)[::4]
hists_annual = hists_norm.resample(time='1Y').mean()
hists_month = hists_norm.resample(time='1M').mean()


# %% load era5 surface temp 
path_t2m = '/pool/data/ERA5/E5/sf/an/1M/167/'
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [f for f in files if int(re.search(r'_(\d{4})_', f).group(1)) >= 2000]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

#  slect tropics and calculate annual average 
with ProgressBar():
    t_month = ds['t2m'].where((ds['latitude'] >= -30) & (ds['latitude'] <= 30)).mean('values').compute()
# 
t_annual = t_month.resample(time='1YE').mean('time')

# %% regress annual histograms on annual mean temperature in every bin
from scipy.stats import linregress

slopes = []
intercepts = []
rms = []

temp_vals = t_annual.sel(time=hists_annual.time).values
for i in range(hists_annual.bin_center.size):
    hist_vals = hists_annual.isel(bin_center=i).values
    res = linregress(temp_vals, hist_vals)
    slopes.append(res.slope)
    intercepts.append(res.intercept)
    rms.append(res.rvalue)

slopes = xr.DataArray(slopes, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])
intercepts = xr.DataArray(intercepts, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])
rms = xr.DataArray(rms, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])


# %% plot interannual varuability and regression slopes
norm = plt.Normalize(vmin=hists_annual.time.dt.year.min(), vmax=hists_annual.time.dt.year.max())
cmap = plt.get_cmap('viridis', len(hists_annual.time))
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex='col', width_ratios=[24, 1])

# distributions
hists_annual = hists_annual - hists_annual.mean('time')
for t in hists_annual.time:
    axes[0, 0].stairs(hists_annual.sel(time=t), bins, alpha=0.7, color=cmap(norm(t.dt.year)))
axes[0, 0].set_ylabel('P(I)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[0, 1], label='Year')


#slopes
axes[1, 0].stairs(slopes, bins, color='red')
axes[1, 0].axhline(0, color='gray', linestyle='-')
axes[1, 0].set_ylabel('dP(I)/dT / K$^{-1}$')
axes[1, 0].set_xlabel('I / kg m$^{-2}$')

for ax in axes[:, 0]:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(1e-3, 3e1)
    ax.set_xscale('log')

axes[1, 1].remove()
fig.tight_layout()

# %% detrend and deseasonalize monthly values

# temperature
t_detrend = xr.DataArray(detrend(t_month), coords=t_month.coords, dims=t_month.dims)
t_deseason = t_detrend.groupby('time.month') - t_detrend.groupby('time.month').mean('time')
t_smooth = t_deseason.rolling(time=3, center=True).mean()
t_smooth['time'] = pd.to_datetime(t_smooth['time'].dt.strftime('%Y-%m'))

# histograms 
hists_anomal = hists_month - hists_month.mean('time')
hists_detrend = xr.DataArray(detrend(hists_anomal, axis=1), coords=hists_anomal.coords, dims=hists_anomal.dims)
hists_deseason = hists_detrend.groupby('time.month') - hists_detrend.groupby('time.month').mean('time')
hists_smooth = hists_deseason.rolling(time=3, center=True).mean().isel(time=slice(1, -1))
hists_smooth['time'] = pd.to_datetime(hists_smooth['time'].dt.strftime('%Y-%m'))

# regression
slopes_m = []
intercepts_m = []
rms_m = []
temp_vals_m = t_smooth.sel(time=hists_smooth.time).values
for i in range(hists_smooth.bin_center.size):
    hist_vals_m = hists_smooth.isel(bin_center=i).values
    res_m = linregress(temp_vals_m, hist_vals_m)
    slopes_m.append(res_m.slope)
    intercepts_m.append(res_m.intercept)
    rms_m.append(res_m.rvalue)
slopes_m = xr.DataArray(slopes_m, coords={'bin_center': hists_smooth.bin_center}, dims=['bin_center'])
intercepts_m = xr.DataArray(intercepts_m, coords={'bin_center': hists_smooth.bin_center}, dims=['bin_center'])
rms_m = xr.DataArray(rms_m, coords={'bin_center': hists_smooth.bin_center}, dims=['bin_center'])

# %% plot monthly variablity and regression slopes
norm = plt.Normalize(vmin=hists_smooth.time.dt.year.min(), vmax=hists_smooth.time.dt.year.max())
cmap = plt.get_cmap('viridis', len(hists_smooth.time))
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex='col', width_ratios=[24, 1])      

# distributions
for t in hists_smooth.time:
    axes[0, 0].stairs(hists_smooth.sel(time=t), bins, alpha=0.7, color=cmap(norm(t.dt.year)))
axes[0, 0].set_ylabel('P(I)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axes[0, 1], label='Year')

#slopes
axes[1, 0].plot(slopes_m.bin_center, slopes_m, color='red')
axes[1, 0].axhline(0, color='gray', linestyle='-')
axes[1, 0].set_ylabel('dP(I)/dT / K$^{-1}$')
axes[1, 0].set_xlabel('I / kg m$^{-2}$')

for ax in axes[:, 0]:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(1e-3, 3e1)
    ax.set_xscale('log')

axes[1, 1].remove()
# %%
