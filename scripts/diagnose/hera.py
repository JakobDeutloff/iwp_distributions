# %%
import intake
#from easygems import healpix as egh

import matplotlib.pyplot as plt
import warnings
from scipy.signal import detrend
import xarray as xr
import glob
import pandas as pd
import re
import cartopy.crs as crs

warnings.filterwarnings("ignore", category=FutureWarning) 
 
 # %% load hera data
current_location = "online"
cat = intake.open_catalog("https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml")[current_location]
ds = cat['ERA5'].to_dask().sel(time=slice('2000-01-01', None))
t_hera = ds['2t'].load()
t_trop_hera = t_hera.where((t_hera.lat >= -30) & (t_hera.lat <= 30)).mean('cell')

# %%
cat = intake.open_catalog(
    "https://gitlab.dkrz.de/data-infrastructure-services/era5-kerchunks/-/raw/main/main.yaml"
)
an = cat["surface_analysis_monthly"].to_dask()
fc = cat["surface_forecast_monthly"].to_dask()
pl = cat["pressure-level_analysis_monthly"].to_dask()

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords").chunk({'time': 1, 'values':-1})
# slect tropics and calculate annual average
t_trop_era = (
    ds["t2m"]
    .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30))
    .mean("values")
    .compute()
)
t_trop_era["time"] = pd.to_datetime(t_trop_era["time"].dt.strftime("%Y-%m"))
# %% detrend and deseasonalize temp 
t_hera_detrend = xr.DataArray(detrend(t_trop_hera), coords=t_trop_hera.coords, dims=t_trop_hera.dims)
t_hera_deseason = t_hera_detrend.groupby('time.month') - t_hera_detrend.groupby('time.month').mean('time')

t_era_detrend = xr.DataArray(detrend(t_trop_era), coords=t_trop_era.coords, dims=t_trop_era.dims)
t_era_deseason = t_era_detrend.groupby('time.month') - t_era_detrend.groupby('time.month').mean('time')

# %% plot both
fig, ax = plt.subplots(figsize=(10, 5))
t_hera_deseason.plot(ax=ax, label='HERA', color='blue')
t_era_deseason.plot(ax=ax, label='ERA5', color='orange')

# %% detrend and deseason whole hera data 
t_hera_detrend_full = xr.DataArray(detrend(t_hera), coords=t_hera.coords, dims=t_hera.dims)
t_hera_deseason_full = t_hera_detrend_full.groupby('time.month') - t_hera_detrend_full.groupby('time.month').mean('time')

# %% select warm years and plot pattern 
t_warm = t_hera_deseason_full.where(t_hera_deseason > 0.4, drop=True)
# %%
for t in t_warm.time:
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
    ax.set_global()
    egh.healpix_show(t_hera.sel(time=t), cmap='coolwarm', ax=ax)
    ax.coastlines()


# %%
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': crs.PlateCarree()})
#ax.set_extent([130, 160, -10, 10], crs=crs.PlateCarree()) 
ax.set_global()
egh.healpix_show(t_warm.mean('time'), cmap='coolwarm', ax=ax)
ax.coastlines()


# %%
t_twp = t_hera.where((t_hera.lat >= -10) & (t_hera.lat <= 10) & (t_hera.lon >= 120) & (t_hera.lon <= 160)).mean('cell')
