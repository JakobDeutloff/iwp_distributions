# %% import packages
import xarray as xr
import numpy as np

# %% load data
ds = xr.open_dataset("data.nc")[["clivi", "qsvi", "qgvi"]]  # only loading ice species is sufficient  
ds_trop = ds.where((ds['clat'] < 30) & (ds['clat'] > -30), drop=True)  # select tropics - you might have to attach coordinates first
iwp = ds_trop['clivi'] + ds_trop['qsvi'] + ds_trop['qgvi']  # calculate iwp

# %% calculate histogram
bins = np.logspace(-3, 2, 254)
hist, _ = np.histogram(iwp, bins=bins, density=False)  # set density=False to get counts instead of probability density
size = iwp.size  # the total number of data points for normalization

# %% construct dataset
bin_midpoints = (bins[1:] + bins[:-1]) / 2
hist_xr = xr.Dataset(
    {
        "hist": (("iwp"), hist),
        "size": size,
    },
    coords={
        "iwp": bin_midpoints,
    },
)
hist_xr.to_netcdf("histogram.nc") 

