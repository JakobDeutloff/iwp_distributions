# %%
import ccic
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% load dardar 
dardar_raw = xr.open_dataset("/work/bm1183/m301049/dardar/dardar_iwp_2008.nc")
mask = (dardar_raw['latitude'] > -30) & (dardar_raw['latitude'] < 30) 
dardar_raw = dardar_raw.where(mask)

# %%
ds = xr.open_zarr('/work/bm1183/m301049/ccic/ccic_cpcir_200801010100.zarr')

# %% load ccic
ds = xr.open_mfdataset("/work/bm1183/m301049/ccic/*.zarr", engine='zarr')

# %% get histograms 
hists = {}
iwp_bins = np.logspace(-4, 2, 100)
hists['dardar'], edges = np.histogram(dardar_raw['iwp']*1e-3, bins=iwp_bins)
hists['ccic'], edges = np.histogram(ds['tiwp'].sel(latitude=slice(30, -30)), bins=iwp_bins)
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.stairs(hists['ccic'], edges, color='red', label='CCIC')
ax.stairs(hists['dardar'], edges, color='blue', label='DARDAR')
ax.set_xlabel('Ice Water Path (kg m$^{-2}$)')
ax.set_ylabel('Counts')
ax.set_xscale('log')
# %%
