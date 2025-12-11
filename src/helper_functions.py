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
    if len(hist.dims) == 2:
        hist_monthly = hist_monthly.transpose("local_time", "time")
    return hist_monthly

def normalise_histograms(hist):
    hist = hist["hist"] / hist["hist"].sum("local_time")
    if len(hist.dims) == 2:
        hist = hist.transpose("local_time", "time")
    return hist


def deseason(ts):
    ts_deseason = ts.groupby("time.month") - ts.groupby("time.month").mean(
        "time"
    )
    ts_deseason["time"] = pd.to_datetime(ts_deseason["time"].dt.strftime("%Y-%m"))
    return ts_deseason

def regress_hist_temp_1d(hist_detrend, temp, hist):
    slopes = []
    err = []
    hist_dummy = hist_detrend.where(hist_detrend.notnull(), drop=True)
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
    mean_hist = hist['hist'].sum('time') / hist['hist'].sum(['time', 'local_time'])
    slopes_perc = slopes_da * 100 / mean_hist
    err_perc = err_da * 100 / mean_hist
    return slopes_perc, err_perc

def detrend_hist_2d(hist):

    out = xr.zeros_like(hist)
    if "bt" in hist.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"
    for i in hist[detrend_dim]:
        hist_detrend = nan_detrend(hist.sel({detrend_dim: i}), dim="local_time")
        out.loc[{detrend_dim: i}] = hist_detrend
    return out

def regress_hist_temp_2d(hist_detrend, temp, hist):
    if "bt" in hist_detrend.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    slopes = xr.zeros_like(hist_detrend.isel(time=0))
    p_values = xr.zeros_like(hist_detrend.isel(time=0))
    for i in hist_detrend.local_time:
        for j in hist_detrend[detrend_dim]:
            hist_vals = hist_detrend.sel({"local_time": i, detrend_dim: j})
            hist_vals = hist_vals.where(np.isfinite(hist_vals), drop=True)
            temp_vals = temp.sel(time=hist_vals.time)
            slope, intercept, r_value, p_value, std_err = linregress(
                temp_vals.values, hist_vals.values
            )
            slopes.loc[{"local_time": i, detrend_dim: j}] = slope
            p_values.loc[{"local_time": i, detrend_dim: j}] = p_value

    mean_hist = hist.mean('time')
    slopes_perc = slopes * 100 / mean_hist
    return slopes_perc, p_values

def lowpass_filter(da, cutoff_period_years=3):
    """
    Apply a lowpass filter using FFT to keep only periods longer than cutoff_period_years.

    Parameters:
    -----------
    da : xarray.DataArray
        Input data array with a 'time' dimension
    cutoff_period_years : float
        Cutoff period in years. Periods longer than this will be kept.

    Returns:
    --------
    xarray.DataArray
        Filtered data array
    """
    # Get time spacing (assuming monthly data)
    time_diff = da.time.diff("time").dt.days.mean().values  # days
    dt = time_diff / 365.25  # convert to years

    # Get number of time steps and find time axis
    n = len(da.time)
    time_axis = da.dims.index("time")

    # Compute FFT along time axis
    fft_data = np.fft.fft(da.values, axis=time_axis)

    # Get frequency array
    freqs = np.fft.fftfreq(n, d=dt)  # frequencies in cycles per year

    # Create filter: keep only frequencies corresponding to periods > cutoff_period_years
    # Period = 1/frequency, so frequency < 1/cutoff_period_years
    cutoff_freq = 1.0 / cutoff_period_years
    filter_mask = np.abs(freqs) < cutoff_freq

    # Apply filter in frequency domain by multiplying with the filter mask
    # Reshape filter_mask to broadcast correctly along all dimensions
    filter_shape = [1] * fft_data.ndim
    filter_shape[time_axis] = n
    filter_mask_broadcast = filter_mask.reshape(filter_shape)
    fft_filtered = fft_data * filter_mask_broadcast

    # Inverse FFT to get filtered time series
    filtered_data = np.fft.ifft(fft_filtered, axis=time_axis).real

    # Create output DataArray with same coordinates
    return xr.DataArray(filtered_data, coords=da.coords, dims=da.dims)