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

def nan_detrend(da, dim='bin_center'):
    out = xr.zeros_like(da)
    for i in da[dim]:
        y = da.sel({dim: i}).values
        mask = np.isfinite(y)
        if np.sum(mask) > 1:
            x = np.arange(len(y))
            # fit linear trend 
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            trend = slope * x + intercept
            out.loc[{dim: i}] = y - trend
    return out



def interpolate_bins(hist, new_bins, name_old_bins):
    """
    Interpolates a histogram defined on old bins to new bins using log-space CDF interpolation.
    Parameters:
    hist (xr.DataArray): The histogram to interpolate.
    new_bins (array-like): The new bin edges to interpolate onto.
    name_old_bins (str): The name of the dimension in hist that corresponds to the old bins.
    Returns:
    xr.DataArray: The interpolated histogram on the new bins.
    """
    cdf = hist.cumsum(name_old_bins)
    cdf[name_old_bins] = np.log10(cdf[name_old_bins])
    cdf_int = cdf.interp({name_old_bins: np.log10(new_bins)}).rename(
        {name_old_bins: "bin_center"}
    )
    pdf_int = cdf_int.diff("bin_center")
    pdf_int["bin_center"] = 10 ** pdf_int["bin_center"]
    return pdf_int

def shift_longitudes(ds, lon_name='longitude'):
    """Shift longitudes from [-180, 180] to [0, 360]"""
    lon_shifted = ds[lon_name].values.copy()
    lon_shifted[ds[lon_name].values < 0] += 360
    ds[lon_name].values = lon_shifted
    return ds