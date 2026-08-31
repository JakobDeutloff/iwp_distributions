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


def nan_detrend(da, dim="iwp"):
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
    cdf[name_old_bins] = np.log10(hist[name_old_bins])
    cdf_int = cdf.interp({name_old_bins: np.log10(new_bins)}).rename(
        {name_old_bins: "iwp"}
    )
    pdf_int = cdf_int.diff("iwp")
    pdf_int["iwp"] = (new_bins[1:] + new_bins[:-1]) / 2
    return pdf_int


def shift_longitudes(ds, lon_name="longitude"):
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
    path = "/work/bu1562/m301049/ccic_daily_cycle/"
    years = range(2000, 2024)
    months = [f"{i:02d}" for i in range(1, 13)]
    hist_list = []
    for year in years:
        for month in months:
            try:
                ds = xr.open_dataset(f"{path}{year}/{filename}{year}{month}.nc")
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
    ts_deseason = ts.groupby("time.month") - ts.groupby("time.month").mean("time")
    ts_deseason["time"] = pd.to_datetime(ts_deseason["time"].dt.strftime("%Y-%m"))
    return ts_deseason


def regress_hist_temp_1d(cf_detrend, temp, cf):
    slopes = []
    err = []
    cf_dummy = cf_detrend.where(cf_detrend.notnull(), drop=True)
    temp_vals = temp.sel(time=cf_dummy.time).values
    for i in range(cf_dummy.local_time.size):
        cf_vals = cf_dummy.isel(local_time=i).values
        slope, intercept, r_value, p_value, std_err = linregress(temp_vals, cf_vals)
        slopes.append(slope)
        err.append(std_err)
    slopes_da = xr.DataArray(
        slopes,
        coords={"local_time": cf_dummy.local_time},
        dims=["local_time"],
    )
    err_da = xr.DataArray(
        err,
        coords={"local_time": cf_dummy.local_time},
        dims=["local_time"],
    )
    mean_cf = cf.mean("time")
    slopes_perc = slopes_da * 100 / mean_cf
    err_perc = err_da * 100 / mean_cf
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


def regress_hist_temp_2d(cf_detrend, temp, cf):
    if "bt" in cf_detrend.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    slopes = xr.zeros_like(cf_detrend.isel(time=0))
    p_values = xr.zeros_like(cf_detrend.isel(time=0))
    for i in cf_detrend.local_time:
        for j in cf_detrend[detrend_dim]:
            cf_vals = cf_detrend.sel({"local_time": i, detrend_dim: j})
            cf_vals = cf_vals.where(np.isfinite(cf_vals), drop=True)
            temp_vals = temp.sel(time=cf_vals.time)
            slope, intercept, r_value, p_value, std_err = linregress(
                temp_vals.values, cf_vals.values
            )
            slopes.loc[{"local_time": i, detrend_dim: j}] = slope
            p_values.loc[{"local_time": i, detrend_dim: j}] = p_value

    mean_hist = cf.mean("time")
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


def read_era5_vars(mode="all"):

    if mode == "all":
        path = "/work/bu1562/m301049/era5/monthly"
        vars = ["t", "p", "rad_tendency", "stability", "subsidence", "convergence"]
        dataarrays = [
            xr.open_dataarray(f"{path}/{var}.nc", chunks={}, decode_timedelta=False)
            for var in vars
        ]
        dataarrays_uni = [
            da.assign_coords(time=dataarrays[0]["time"]) for da in dataarrays
        ]
        ds_merged = xr.merge(dataarrays_uni, compat="override")
    else:
        path = "/work/bu1562/m301049/era5/monthly/averages"
        vars = ["t", "p", "rad", "stability", "subsidence", "convergence"]
        dataarrays = [
            xr.open_dataarray(
                f"{path}/{var}_mean.nc", chunks={}, decode_timedelta=False
            )
            for var in vars
        ]
        dataarrays_uni = [
            da.assign_coords(time=dataarrays[0]["time"]) for da in dataarrays
        ]
        ds_merged = xr.merge(dataarrays_uni, compat="override")
    return ds_merged


def calculate_jj_mean(ds):
    ds_spring = ds.sel(time=ds["time.month"] < 7).groupby("time.year").mean(dim="time")
    ds_spring = ds_spring.isel(year=slice(1, None))  # remove first year
    ds_spring["year"] = (
        ds_spring["year"] - 1
    )  # shift year to starting year of july-june period
    ds_fall = ds.sel(time=ds["time.month"] >= 7).groupby("time.year").mean(dim="time")
    ds_jj = (
        xr.concat([ds_spring, ds_fall], dim="year")
        .sortby("year")
        .groupby("year")
        .mean(dim="year")
    )
    return ds_jj


def calculate_jj_sum(hist):
    hist_spring = (
        hist.sel(time=hist["time.month"] < 7).groupby("time.year").sum(dim="time")
    )
    hist_spring = hist_spring.isel(year=slice(1, None))  # remove first year
    hist_spring["year"] = (
        hist_spring["year"] - 1
    )  # shift year to starting year of july-june period
    hist_fall = (
        hist.sel(time=hist["time.month"] >= 7).groupby("time.year").sum(dim="time")
    )
    hist_jj = (
        xr.concat([hist_spring, hist_fall], dim="year")
        .sortby("year")
        .groupby("year")
        .sum(dim="year")
    )
    return hist_jj


def load_histograms(set="all", freq="1ME"):

    # observations
    hists_obs = {}
    hist_2d = xr.open_dataset(
        "/work/bu1562/m301049/diurnal_cycle_dists/ccic_2d_monthly_all_weighted_noz.nc"
    )
    hists_obs["ccic"] = hist_2d.sum("local_time")
    hists_obs["two_c_ice"] = xr.open_dataset(
        "/work/bu1562/m301049/cloudsat/dists_no_dup_fine.nc"
    )
    hists_obs["two_c_ice"] = (
        hists_obs["two_c_ice"].coarsen(bin_center=4, boundary="trim").sum()
    )
    hists_obs["dardar"] = xr.open_dataset(
        "/work/bu1562/m301049/dardarv3.10/hist_dardar_fine.nc"
    )
    hists_obs["dardar"] = (
        hists_obs["dardar"].coarsen(bin_center=4, boundary="trim").sum()
    )
    hists_obs["spare_ice"] = xr.open_dataset(
        "/work/bu1562/m301049/spareice/hists_metop_fine.nc"
    ).sel(time=slice(None, "2025-07"))
    hists_obs["spare_ice"] = (
        hists_obs["spare_ice"].coarsen(bin_center=4, boundary="trim").sum()
    )
    for key in hists_obs.keys():
        hists_obs[key] = hists_obs[key].resample(time=freq).sum()
        if "bin_center" in hists_obs[key].dims:
            hists_obs[key] = hists_obs[key].rename({"bin_center": "iwp"})
        hists_obs[key] = hists_obs[key].transpose("time", "iwp")
        hists_obs[key]["time"] = pd.to_datetime(
            hists_obs[key]["time"].dt.strftime("%Y-%m")
        )

    # icon AP
    hists_model = {}
    names = {
        "icon_ap_control": "control",
        "icon_ap_plus4K": "plus4K",
        "icon_ap_plus2K": "plus2K",
    }
    for name, run in names.items():
        hists_model[name] = xr.open_dataset(
            f"/work/bu1562/m301049/icon_hcap_data/{run}/production/daily_cycle_hist_weighted.nc"
        )
        hists_model[name] = hists_model[name].sum(["local_time", "time"])
        hists_model[name] = hists_model[name].coarsen(iwp=4, boundary="trim").sum()
        hists_model[name] = hists_model[name]["hist"] / hists_model[name]["size"]

    # icon AMIP
    icon_amip_cont = (
        xr.open_dataset(
            "/work/bu1562/m301049/icon-amip/histogram_iwp_ctrl_20200401_20200831-2.nc"
        )
        .sel(domain="land_ocean")
        .rename({"iwp_bin": "iwp"})
        .drop_vars("domain")
    )
    icon_amip_cont = icon_amip_cont["pdf"] / (
        icon_amip_cont["pdf"].sum("iwp") + icon_amip_cont["clear_sky_area"]
    )
    icon_amip_4k = (
        xr.open_dataset(
            "/work/bu1562/m301049/icon-amip/histogram_iwp_sst4k_20200401_20200831-2.nc"
        )
        .sel(domain="land_ocean")
        .rename({"iwp_bin": "iwp"})
        .drop_vars("domain")
    )
    icon_amip_4k = icon_amip_4k["pdf"] / (
        icon_amip_4k["pdf"].sum("iwp") + icon_amip_4k["clear_sky_area"]
    )
    hists_model["icon_amip_control"] = icon_amip_cont.coarsen(
        iwp=4, boundary="trim"
    ).sum()
    hists_model["icon_amip_plus4K"] = icon_amip_4k.coarsen(iwp=4, boundary="trim").sum()

    # rcemip
    ds = xr.open_dataset(
        "/work/bu1562/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
    )
    ds["fwp"] = ds["fwp"] * 1e-3
    rcemip_pdf = interpolate_bins(
        ds["f"].mean("model"), np.logspace(-3, 2, 254)[::4], "fwp"
    )
    hists_model["rcemip_control"] = rcemip_pdf.sel(SST=295)
    hists_model["rcemip_plus10K"] = rcemip_pdf.sel(SST=305)

    # xshield
    xshield_cont = xr.open_dataset(
        "/work/bu1562/m301049/xshield/xshield24v2_iw_histogram.nc"
    )
    xshield_4k = xr.open_dataset(
        "/work/bu1562/m301049/xshield/xshield24v2_PLUS_4K_iw_histogram.nc"
    )
    hists_model["xshield_control"] = xshield_cont["f"]
    hists_model["xshield_plus4K"] = xshield_4k["f"]

    # unify iwp axis
    for key in hists_model.keys():
        hists_model[key]["iwp"] = hists_obs["dardar"]["iwp"]
    for key in hists_obs.keys():
        hists_obs[key]["iwp"] = hists_obs["dardar"]["iwp"]

    # package them as xarray datasets and unify
    hists = {}
    if set == "all":
        hists_obs["two_c_ice"] = hists_obs["two_c_ice"].where(
            hists_obs["two_c_ice"]["size"] > 1.9e6
        )
        hists_obs["dardar"] = hists_obs["dardar"].where(
            hists_obs["dardar"]["size"] > 1.9e6
        )
        for key in hists_obs.keys():
            hists[key] = hists_obs[key]["hist"] / hists_obs[key]["size"]
        for key in hists_model.keys():
            hists[key] = hists_model[key]
        return hists
    elif set == "obs":
        for key in hists_obs.keys():
            hists[key] = hists_obs[key]
        return hists
    elif set == "model":
        for key in hists_model.keys():
            hists[key] = hists_model[key]
        return hists
    else:
        raise ValueError("Invalid set specified. Choose from 'all', 'obs', or 'model'.")


def load_slopes():
    slopes = xr.open_dataset("/work/bu1562/m301049/iwp_dists/slopes_monthly.nc")
    errors = xr.open_dataset("/work/bu1562/m301049/iwp_dists/errors_monthly.nc")
    p_vals = xr.open_dataset("/work/bu1562/m301049/iwp_dists/p_vals_monthly.nc")
    return slopes, errors, p_vals


def load_cre():
    cre = xr.open_dataset(
        f"/work/bu1562/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
    )
    hists = load_histograms("obs")
    cre["iwp"] = np.log10(cre["iwp"])
    cre = cre.interp(iwp=np.log10(hists["ccic"].iwp), method="linear").drop_vars("iwp")
    cre["iwp"] = hists["ccic"].iwp
    return cre


def load_feedbacks():
    feedback = xr.open_dataset("/work/bu1562/m301049/iwp_dists/feedback.nc")
    feedback_area = xr.open_dataset("/work/bu1562/m301049/iwp_dists/feedback_area.nc")
    feedback_opacity = xr.open_dataset(
        "/work/bu1562/m301049/iwp_dists/feedback_opacity.nc"
    )
    return feedback, feedback_area, feedback_opacity


runs = ["jed0011", "jed0022", "jed0033"]
experiments = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}


def load_random_datasets(version="processed"):
    """
    Load the random datasets for the model.

    Parameters
    ----------
    processed : bool, optional
        If True, load the processed datasets, otherwise load the raw datasets. Default is True.

    Returns
    -------
    dict
        Dictionary containing the random datasets.
    """
    datasets = {}
    if version == "processed":
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bu1562/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample_processed_64.nc"
            )
    elif version == "temp":
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bu1562/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample_tgrid_20.nc"
            )
    else:
        for run in runs:
            datasets[run] = xr.open_dataset(
                f"/work/bu1562/m301049/icon_hcap_data/{experiments[run]}/production/random_sample/{run}_randsample.nc"
            )
    return datasets
