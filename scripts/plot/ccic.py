# %%
import ccic
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import s3fs

# %% load dardar 
dardar_raw = xr.open_dataset("/work/bm1183/m301049/dardar/dardar_iwp_2008.nc")
mask = (dardar_raw['latitude'] > -20) & (dardar_raw['latitude'] < 20) 
dardar_raw = dardar_raw.where(mask)

# %% load ccic
ds = xr.open_mfdataset("/work/bm1183/m301049/ccic/*.zarr", engine='zarr')

# %% get histograms 
hists = {}
iwp_bins = np.logspace(-4, np.log10(40), 100)
hists['dardar'], edges = np.histogram(dardar_raw['iwp']*1e-3, bins=iwp_bins, density=False)
hists['dardar'] = hists['dardar'] / dardar_raw['iwp'].count().values  # Normalize histogram
hists['ccic'], edges = np.histogram(ds['tiwp'].sel(latitude=slice(20, -20)), bins=iwp_bins, density=False)
hists['ccic'] = hists['ccic'] /  ds['tiwp'].sel(latitude=slice(20, -20)).size
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.stairs(hists['ccic'], edges, color='red', label='CCIC CPCIR')
ax.stairs(hists['dardar'], edges, color='blue', label='DARDAR v2.1')
ax.set_xlabel('Ice Water Path (kg m$^{-2}$)')
ax.set_ylabel('Counts')
ax.set_xscale('log')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
fig.savefig('plots/ccic_histogram.png', bbox_inches='tight', dpi=300)
# %%
