# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from dask.diagnostics import ProgressBar

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

# %% plot interannual variability
fig, ax = plt.subplots(figsize=(12, 6))
norm = plt.Normalize(vmin=hists_annual.time.dt.year.min(), vmax=hists_annual.time.dt.year.max())
cmap = plt.get_cmap('viridis', len(hists_annual.time))
for t in hists_annual.time:
    ax.stairs(hists_annual.sel(time=t), bins, alpha=0.7, color=cmap(norm(t.dt.year)))
ax.set_xscale('log')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Year')
ax.set_xlabel('I / kg m$^{-2}$')
ax.set_ylabel('P(I)')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(1e-3, 3e1)


# %% load era5 surface temp 
path_t2m = '/pool/data/ERA5/E5/sf/an/1M/167/'
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [f for f in files if int(re.search(r'_(\d{4})_', f).group(1)) >= 2000]

# %% Open multiple files as a single dataset
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

# %% slect tropics and calculate annual average 
with ProgressBar():
    t_tropics = ds['t2m'].where((ds['latitude'] >= -30) & (ds['latitude'] <= 30)).mean('values').compute()

# %% 
temp_annual = t_tropics.resample(time='1YE').mean('time')

# %% regress annual histograms on annual mean temperature in every bin
from scipy.stats import linregress

slopes = []
intercepts = []
rms = []

temp_vals = temp_annual.sel(time=hists_annual.time).values
for i in range(hists_annual.bin_center.size):
    hist_vals = hists_annual.isel(bin_center=i).values
    res = linregress(temp_vals, hist_vals)
    slopes.append(res.slope)
    intercepts.append(res.intercept)
    rms.append(res.rvalue)

slopes = xr.DataArray(slopes, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])
intercepts = xr.DataArray(intercepts, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])
rms = xr.DataArray(rms, coords={'bin_center': hists_annual.bin_center}, dims=['bin_center'])


# %% plot slopes 
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(slopes.bin_center, slopes, color='red')
ax.set_xscale('log')
ax.axhline(0, color='black', linestyle='--', alpha=0.7)
ax.set_xlabel('I / kg m$^{-2}$')
ax.set_ylabel('dP(I)/dT / K$^{-1}$')

# %%
