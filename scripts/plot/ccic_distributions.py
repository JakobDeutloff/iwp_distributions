# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress

# %% initialize containers
hists_monthly = {}
hists_annual = {}
hists_smooth = {}
slopes_monthly = {}
slopes_annual = {}
error_montly = {}
error_annual = {}
bins = bins = np.logspace(-3, 2, 254)[::4]

# %% open distributions CCIC
path = "/work/bm1183/m301049/ccic/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hists_ccic = xr.concat(hist_list, dim="time")

# %% open distributions 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% coarsen histograms and normalise by size
hists_coarse = hists_ccic.coarsen(bin_center=4, boundary="trim").sum()
hists_norm = hists_coarse["hist"] / hists_coarse["size"]

hists_annual["ccic"] = hists_norm.resample(time="1Y").mean()
hists_monthly["ccic"] = hists_norm.resample(time="1M").mean()
hists_monthly["ccic"]["time"] = pd.to_datetime(
    hists_monthly["ccic"]["time"].dt.strftime("%Y-%m")
)

hists_monthly["2c"] = (hists_2c["hist"] / hists_2c["size"]).sel(
    time=slice("2006-06", "2017-12")
)
good_years = ["2007", "2008", "2009", "2010", "2013", "2014", "2015", "2016"]
hists_2c_annual = hists_2c.resample(time="1YE").mean()
mask = np.isin(hists_2c_annual["time"].dt.year.astype(str), good_years)
mask_good_years = xr.DataArray(
    mask, dims=["time"], coords={"time": hists_2c_annual["time"]}
)
hists_2c_annual = hists_2c_annual.where(mask_good_years)
hists_annual["2c"] = (hists_2c_annual["hist"] / hists_2c_annual["size"]).transpose()

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

#  slect tropics and calculate annual average
with ProgressBar():
    t_month = (
        ds["t2m"]
        .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30))
        .mean("values")
        .compute()
    )

t_month["time"] = pd.to_datetime(t_month["time"].dt.strftime("%Y-%m"))
t_annual = t_month.resample(time="1YE").mean("time")

# %% regress annual histograms on annual mean temperature in every bin
from scipy.stats import linregress

slopes_ccic = []
err_ccic = []

temp_ccic = t_annual.sel(time=hists_annual["ccic"].time)
for i in range(hists_annual["ccic"].bin_center.size):
    hist_vals = hists_annual["ccic"].isel(bin_center=i).values
    res = linregress(temp_ccic.values, hist_vals)
    slopes_ccic.append(res.slope)
    err_ccic.append(res.stderr)

slopes_annual["ccic"] = xr.DataArray(
    slopes_ccic,
    coords={"bin_center": hists_annual["ccic"].bin_center},
    dims=["bin_center"],
)
error_annual["ccic"] = xr.DataArray(
    err_ccic,
    coords={"bin_center": hists_annual["ccic"].bin_center},
    dims=["bin_center"],
)


hists_dummy = hists_annual["2c"].where(mask_good_years, drop=True)
temp_2c = t_annual.sel(time=hists_dummy.time)
slopes_2c = []
err_2c = []
for i in range(hists_dummy.bin_center.size):
    hist_vals_2c = hists_dummy.isel(bin_center=i).values
    res_2c = linregress(temp_2c.values, hist_vals_2c)
    slopes_2c.append(res_2c.slope)
    err_2c.append(res_2c.stderr)

slopes_annual["2c"] = xr.DataArray(
    slopes_2c, coords={"bin_center": hists_annual["2c"].bin_center}, dims=["bin_center"]
)
error_annual["2c"] = xr.DataArray(
    err_2c, coords={"bin_center": hists_annual["2c"].bin_center}, dims=["bin_center"]
)


# %% plot annual variability and regression slopes


def plot_regression(temp, hists, slopes, error, title):
    fig, axes = plt.subplots(
        2, 3, figsize=(8, 8), sharex="col", width_ratios=[6, 24, 1]
    )

    # temp
    axes[0, 0].plot(temp.values, temp.time, color="black")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("T / K")
    axes[0, 0].set_ylabel("Year")
    axes[0, 0].set_ylim([hists.time.max(), hists.time.min()])
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    # iwp anomalies
    im = axes[0, 1].pcolormesh(
        hists.bin_center,
        hists.time,
        (hists - hists.mean("time")).T,
        cmap="seismic",
        vmin=-0.001,
        vmax=0.001,
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlim([1e-3, 2e1])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_ylim([hists.time.max(), hists.time.min()])
    axes[0, 1].set_yticklabels([])

    # slopes
    axes[1, 1].plot(hists.bin_center, slopes, color="k", label="CCIC")
    axes[1, 1].fill_between(
        hists.bin_center,
        slopes - error,
        slopes + error,
        color="gray",
        alpha=0.5,
        label="95% confidence interval",
    )
    axes[1, 1].axhline(0, color="gray", linestyle="-")
    axes[1, 1].set_ylabel("dP(I)/dT / K$^{-1}$")
    axes[1, 1].set_xlabel("I / kg m$^{-2}$")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    axes[1, 0].remove()
    axes[1, 2].remove()
    fig.colorbar(im, cax=axes[0, 2], label="P(I) anomaly")

    fig.suptitle(title, y=0.95)
    return fig, axes


def plot_hists(hists, temp):
    temp = temp.sel(time=hists.time)
    norm = plt.Normalize(vmin=temp.min(), vmax=temp.max())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for t in hists.time:
        axes[0].stairs(
            hists.sel(time=t),
            bins,
            alpha=0.7,
            color=cmap(norm(temp.sel(time=t).values)),
        )
        axes[1].stairs(
            hists.sel(time=t) - hists.mean("time"),
            bins,
            alpha=0.7,
            color=cmap(norm(temp.sel(time=t).values)),
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim([1e-3, 2e1])
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("P(I)")
    axes[1].set_ylabel("P(I) anomaly")
    axes[1].set_xlabel("I / kg m$^{-2}$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Temperature / K")
    return fig


# %%
fig_ccic, axes_ccic = plot_regression(
    temp_ccic,
    hists_annual["ccic"],
    slopes_annual["ccic"],
    error_annual["ccic"],
    "CCIC Annual",
)
fig_ccic.savefig("plots/ccic_annual.png", dpi=300, bbox_inches="tight")

# %%
fig_2c, axes_2c = plot_regression(
    temp_2c,
    hists_annual["2c"],
    slopes_annual["2c"],
    error_annual["2c"],
    "2C-ICE Annual",
)
fig_2c.savefig("plots/2c_annual.png", dpi=300, bbox_inches="tight")

# %%
plot_hists(hists_monthly["ccic"], t_month)

# %%
fig = plot_hists(hists_annual["2c"].sel(time=slice(None, "2011-01")), t_annual)

# %% detrend and deseasonalize monthly values

# temperature
t_detrend = xr.DataArray(detrend(t_month), coords=t_month.coords, dims=t_month.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)
t_smooth = t_deseason.rolling(time=3, center=True).mean()
t_smooth["time"] = pd.to_datetime(t_smooth["time"].dt.strftime("%Y-%m"))

# histograms ccic
hists_detrend = xr.DataArray(
    detrend(hists_monthly["ccic"], axis=1),
    coords=hists_monthly["ccic"].coords,
    dims=hists_monthly["ccic"].dims,
)
hists_deseason = hists_detrend.groupby("time.month") - hists_detrend.groupby(
    "time.month"
).mean("time")
hists_deseason["time"] = pd.to_datetime(hists_deseason["time"].dt.strftime("%Y-%m"))
hists_smooth_ccic = (
    hists_deseason.rolling(time=3, center=True).mean().isel(time=slice(1, -1))
)
hists_smooth_ccic["time"] = pd.to_datetime(
    hists_smooth_ccic["time"].dt.strftime("%Y-%m")
)
hists_smooth["ccic"] = hists_smooth_ccic

# histograms 2c-ice
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


hists_2c_detrend = nan_detrend_along_time(
    hists_monthly["2c"].sel(time=slice("2007-01", "2017-01"))
)
hists_2c_deseason = hists_2c_detrend.groupby("time.month") - hists_2c_detrend.groupby(
    "time.month"
).mean("time")
hists_smooth_2c = (
    hists_2c_deseason.rolling(time=3, center=True, min_periods=1)
    .mean()
    .where(hists_2c_detrend.notnull())
)
hists_smooth_2c["time"] = pd.to_datetime(hists_smooth_2c["time"].dt.strftime("%Y-%m"))
hists_smooth["2c"] = hists_smooth_2c

# %%regression
slopes_ccic = []
err_ccic = []
temp_vals_ccic = t_month.sel(time=hists_deseason.time).values
for i in range(hists_deseason.bin_center.size):
    hist_vals = hists_deseason.isel(bin_center=i).values
    res = linregress(temp_vals_ccic, hist_vals)
    slopes_ccic.append(res.slope)
    err_ccic.append(res.stderr)

slopes_monthly["ccic"] = xr.DataArray(
    slopes_ccic, coords={"bin_center": hists_deseason.bin_center}, dims=["bin_center"]
)
error_montly["ccic"] = xr.DataArray(
    err_ccic, coords={"bin_center": hists_deseason.bin_center}, dims=["bin_center"]
)

slopes_2c = []
err_2c = []
hists_dummy = hists_smooth["2c"].where(hists_2c_deseason.notnull(), drop=True)
temp_vals_2c = t_smooth.sel(time=hists_dummy.time).values
for i in range(hists_dummy.bin_center.size):
    hist_vals = hists_dummy.isel(bin_center=i).values
    res = linregress(temp_vals_2c, hist_vals)
    slopes_2c.append(res.slope)
    err_2c.append(res.stderr)

slopes_monthly["2c"] = xr.DataArray(
    slopes_2c, coords={"bin_center": hists_smooth["2c"].bin_center}, dims=["bin_center"]
)
error_montly["2c"] = xr.DataArray(
    err_2c, coords={"bin_center": hists_smooth["2c"].bin_center}, dims=["bin_center"]
)

# %%
fig, axes = plot_regression(
    t_smooth,
    hists_deseason,
    slopes_monthly["ccic"],
    error_montly["ccic"],
    "CCIC Monthly",
)
fig.savefig("plots/ccic_monthly.png", dpi=300, bbox_inches="tight")
# %%
fig, axes = plot_regression(
    t_smooth,
    hists_smooth["2c"],
    slopes_monthly["2c"],
    error_montly["2c"],
    "2C-ICE Monthly",
)
fig.savefig("plots/2c_monthly.png", dpi=300, bbox_inches="tight")
# %% check detrending and deseasonalising
ts = hists_monthly["2c"].isel(bin_center=12)
ts = (ts - ts.mean("time")).where(ts.notnull(), drop=True)
ts_detrend = xr.DataArray(detrend(ts), coords=ts.coords, dims=ts.dims)
ts_deseason = ts_detrend.groupby("time.month") - ts_detrend.groupby("time.month").mean(
    "time"
)
ts_smooth = ts_deseason.rolling(time=3, center=True).mean().isel(time=slice(1, -1))
ts_alt = hists_2c_deseason.isel(bin_center=12)

fig, ax = plt.subplots()
ax.plot(ts, label="original")
ax.plot(ts_detrend, label="detrended")
ax.plot(ts_deseason, label="deseasonalized")
ax.plot(ts_smooth, label="smoothed")
ax.plot(ts_alt, label="from smoothed histograms")
ax.legend()

print("Mean original:", ts.mean().item())
print("Mean detrended:", ts_detrend.mean().item())
print("Mean deseasonalized:", ts_deseason.mean().item())
print("Mean smoothed:", ts_smooth.mean().item())
# %%
