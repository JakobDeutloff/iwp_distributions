import numpy as np
import xarray as xr 
from scipy.signal import detrend

def nan_detrend_along_time(da):
    arr = da.values
    out = np.full_like(arr, np.nan)
    # Detrend each bin (column) separately
    for i in range(arr.shape[0]):
        y = arr[i, :]
        mask = np.isfinite(y)
        if np.sum(mask) > 1:
            y_detrended = detrend(y[mask])
            out[i, mask] = y_detrended
    return xr.DataArray(out, coords=da.coords, dims=da.dims)

def nan_detrend(da):
    out = np.full_like(da.values, np.nan)
    for i in range(da.shape[0]):
        y = da[i, :].values
        mask = np.isfinite(y)
        if np.sum(mask) > 1:
            x = np.arange(len(y))
            # fit linear trend 
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            trend = slope * x + intercept
            out[i, :] = y - trend
    return xr.DataArray(out, coords=da.coords, dims=da.dims)



