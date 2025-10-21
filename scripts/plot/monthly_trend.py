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
from src.plot import plot_regression, plot_hists
from src.helper_functions import nan_detrend_along_time

# %% initialize containers
hists_monthly = {}
hists_smooth = {}
slopes_monthly = {}
error_montly = {}
bins = bins = np.logspace(-3, 2, 254)[::4]
datasets = ["ccic", "2c", "dardar"]

# %% open CCIC
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

# %% open 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% open dardar
hists_dardar = xr.open_dataset("/work/bm1183/m301049/dardarv3.10/hist_dardar.nc")

# %% coarsen histograms and normalise by size
hists_ccic = hists_ccic.coarsen(bin_center=4, boundary="trim").sum()

# ccic
hists_monthly["ccic"] = (
    hists_ccic["hist"].resample(time="1ME").sum()
    / hists_ccic["size"].resample(time="1ME").sum()
)

# 2c-ice
hists_monthly["2c"] = (hists_2c["hist"] / hists_2c["size"]).sel(
    time=slice("2006-06", "2017-12")
)

# dardar
hists_monthly["dardar"] = (
    hists_dardar["hist"].resample(time="1ME").sum()
    / hists_dardar["size"].resample(time="1ME").sum()
)

# %% fix time coordinates
for key in datasets:
    hists_monthly[key]["time"] = pd.to_datetime(
        hists_monthly[key]["time"].dt.strftime("%Y-%m")
    )

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

# %%
plot_hists(hists_monthly["dardar"], t_month, bins)

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

# histograms dardar
hists_dardar_detrend = nan_detrend_along_time(
    hists_monthly["dardar"]
)
hists_dardar_deseason = hists_dardar_detrend.groupby("time.month") - hists_dardar_detrend.groupby(
    "time.month"
).mean("time")
hists_smooth_dardar = (
    hists_dardar_deseason.rolling(time=3, center=True, min_periods=1)
    .mean()
    .where(hists_dardar_detrend.notnull())
)
hists_smooth['dardar'] = hists_smooth_dardar

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

slopes_dardar = []
err_dardar = []
hists_dummy = hists_smooth["dardar"].where(hists_dardar_deseason.notnull(), drop=True)
temp_vals_dardar = t_smooth.sel(time=hists_dummy.time).values
for i in range(hists_dummy.bin_center.size):
    hist_vals = hists_dummy.isel(bin_center=i).values
    res = linregress(temp_vals_dardar, hist_vals)
    slopes_dardar.append(res.slope)
    err_dardar.append(res.stderr)

slopes_monthly["dardar"] = xr.DataArray(
    slopes_dardar, coords={"bin_center": hists_smooth["dardar"].bin_center}, dims=["bin_center"]
)
error_montly["dardar"] = xr.DataArray(
    err_dardar, coords={"bin_center": hists_smooth["dardar"].bin_center}, dims=["bin_center"]
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

# %% 
fig, axes = plot_regression(
    t_smooth,
    hists_smooth["dardar"],
    slopes_monthly["dardar"],
    error_montly["dardar"],
    "DARDAR v3.10 Monthly",
)
fig.savefig("plots/dardar_monthly.png", dpi=300, bbox_inches="tight")
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
