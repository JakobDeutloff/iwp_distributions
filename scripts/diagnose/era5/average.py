# %%
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar

dask.config.set(scheduler='synchronous')

# %%
path = "/work/bm1183/m301049/era5/monthly"
datasets= {
    't': xr.open_dataarray(f"{path}/t.nc", chunks={}, decode_timedelta=False),
    'p': xr.open_dataarray(f"{path}/p.nc", chunks={}, decode_timedelta=False),
    'rad': xr.open_dataarray(f"{path}/rad_tendency.nc", chunks={}, decode_timedelta=False),
    'stability': xr.open_dataarray(f"{path}/stability.nc", chunks={}, decode_timedelta=False),
    'subsidence': xr.open_dataarray(f"{path}/subsidence.nc", chunks={}, decode_timedelta=False),
    'convergence': xr.open_dataarray(f"{path}/convergence.nc", chunks={}, decode_timedelta=False),
}



# %% merge into one xarray 
xr_dataset = xr.merge(datasets.values(), compat='override')
xr_dataset.to_netcdf(f"{path}/era5_stability_all_vars.nc")
#%% latitude weighted mean 
for key, da in datasets.items():
    weights = np.cos(np.deg2rad(da.latitude))    
    weighted_mean = da.weighted(weights).mean(dim=['latitude', 'longitude'])
    print(f'save weighted mean for {key}')
    with ProgressBar():
        weighted_mean.to_netcdf(f"{path}/averages/{key}_mean.nc")

