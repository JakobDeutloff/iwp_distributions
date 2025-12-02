import numpy as np
import xarray as xr 
from scipy.signal import detrend
import pandas as pd
from scipy.stats import linregress

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
    lon_shifted[ds[lon_name].values < 0] += 360.0
    if lon_name in ds.dims:
        ds = ds.assign_coords({lon_name: lon_shifted})
        ds = ds.sortby(lon_name)
    else:
        ds[lon_name].values = lon_shifted
    return ds

def read_ccic_dc(filename):
    path = "/work/bm1183/m301049/ccic_daily_cycle/"
    years = range(2000, 2024)
    months = [f"{i:02d}" for i in range(1, 13)]
    hist_list = []
    for year in years:
        for month in months:
            try:
                ds = xr.open_dataset(
                    f"{path}{year}/{filename}{year}{month}.nc"
                )
                hist_list.append(ds)
            except FileNotFoundError:
                print(f"File for {year}-{month} not found, skipping.")

    hists_ccic = xr.concat(hist_list, dim="time")
    return hists_ccic

def resample_histograms(hist):
    hist_monthly = hist.resample(time="1ME").sum()
    hist_monthly["time"] = pd.to_datetime(hist_monthly["time"].dt.strftime("%Y-%m"))
    hist_monthly = hist_monthly["hist"] / hist_monthly["hist"].sum("local_time")
    hist_monthly = hist_monthly.transpose("local_time", "time")
    return hist_monthly

def deseason(ts):
    ts_deseason = ts.groupby("time.month") - ts.groupby("time.month").mean(
        "time"
    )
    ts_deseason["time"] = pd.to_datetime(ts_deseason["time"].dt.strftime("%Y-%m"))
    return ts_deseason

def regress_hist_temp_1d(hist, temp):
    slopes = []
    err = []
    hist_dummy = hist.where(hist.notnull(), drop=True)
    temp_vals = temp.sel(time=hist_dummy.time).values
    for i in range(hist_dummy.local_time.size):
        hist_vals = hist_dummy.isel(local_time=i).values
        slope, intercept, r_value, p_value, std_err = linregress(temp_vals, hist_vals)
        slopes.append(slope)
        err.append(std_err)
    slopes_da = xr.DataArray(
        slopes,
        coords={"local_time": hist_dummy.local_time},
        dims=["local_time"],
    )
    err_da = xr.DataArray(
        err,
        coords={"local_time": hist_dummy.local_time},
        dims=["local_time"],
    )
    return slopes_da, err_da
