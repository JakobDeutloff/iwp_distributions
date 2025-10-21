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