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
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, shift_longitudes
import pickle

# %% initialize containers
hists_monthly = {}
hists_smooth = {}
slopes_monthly = {}
error_montly = {}
bins = bins = np.logspace(-3, 2, 254)[::4]
datasets = ["2c", "dardar"]
colors, line_labels, linestyles = definitions()
lon_min_twp = 120
lon_max_twp = 180

# %% load data
hist_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists_sea.nc")
hist_dardar = xr.open_dataset("/work/bm1183/m301049/dardarv3.10/hist_dardar_sea.nc")
hists_monthly["2c"] = hist_2c
hists_monthly["dardar"] = hist_dardar.resample(time="1ME").sum()
mask = (
    xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")
    .load()
    .pipe(shift_longitudes, lon_name="lon")
)

# %% cut data
for dataset in datasets:
    hists_monthly[dataset]["time"] = pd.to_datetime(
        hists_monthly[dataset]["time"].dt.strftime("%Y-%m")
    )
    hists_monthly[dataset] = hists_monthly[dataset].sel(
        time=slice("2006-07", "2017-12")
    )
    # hists_monthly[dataset] = hists_monthly[dataset].where(
    #     (hists_monthly[dataset].time < pd.to_datetime("2011-01"))
    #     | (hists_monthly[dataset].time > pd.to_datetime("2012-04"))
    # )




# %% normalise data
hists_norm = {}
for dataset in datasets:
    hists_norm[dataset] = (
        hists_monthly[dataset]["hist"] / hists_monthly[dataset]["size"]
    )

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords").chunk(
    {"time": 1, "values": -1}
)
mask = mask.sel(
    lat=ds.isel(time=0).latitude, lon=ds.isel(time=0).longitude, method="nearest"
)
with ProgressBar():
    t_month = (
        ds["t2m"]
        .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30) & mask)
        .mean("values")
        .compute()
    )

t_month["time"] = pd.to_datetime(t_month["time"].dt.strftime("%Y-%m"))

# %%
plot_hists(hists_norm["dardar"], t_month, bins)

# %% filter outliers
for dataset in datasets:
    rms = np.abs(hists_norm[dataset] - hists_norm[dataset].median("time")).sum(
        "bin_center"
    )
    hists_norm[dataset] = hists_norm[dataset].where(
        rms < rms.quantile(0.95)
    )

# %% filter size 
for dataset in datasets:
    hists_norm[dataset] = hists_norm[dataset].where(
        hists_monthly[dataset]['size'] > 1.2e6
    )
# %% detrend and deseasonalize monthly values
hists_deseason = {}

# temperature
t_detrend = xr.DataArray(detrend(t_month), coords=t_month.coords, dims=t_month.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)
for ds in datasets:
    hists_detrend = nan_detrend(hists_norm[ds])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[ds] = hists_deseason_ds

# %%regression
for ds in datasets:
    slopes_ds = []
    err_ds = []
    hist_vals = hists_deseason[ds].where(hists_deseason[ds].notnull(), drop=True)
    temp = t_deseason.sel(time=hist_vals.time)
    for i in range(hists_deseason[ds].bin_center.size):
        hist_row = hist_vals.isel(bin_center=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
    slopes_monthly[ds] = xr.DataArray(
        slopes_ds,
        coords={"bin_center": hists_deseason[ds].bin_center},
        dims=["bin_center"],
    )
    error_montly[ds] = xr.DataArray(
        err_ds,
        coords={"bin_center": hists_deseason[ds].bin_center},
        dims=["bin_center"],
    )

# %%
fig, axes = plot_regression(
    t_deseason.sel(time=hists_deseason["2c"].time),
    hists_deseason["2c"],
    slopes_monthly["2c"],
    error_montly["2c"],
    "2C-ICE Monthly",
)
fig.savefig("plots/2c_monthly_sea.png", dpi=300, bbox_inches="tight")
# %%
fig, axes = plot_regression(
    t_deseason.sel(time=hists_deseason["dardar"].time),
    hists_deseason["dardar"].T,
    slopes_monthly["dardar"],
    error_montly["dardar"],
    "DARDAR Monthly",
)
fig.savefig("plots/dardar_monthly_sea.png", dpi=300, bbox_inches="tight")
# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

t_deseason.plot(ax=axes[0], color="blue", label="TWP")
t_trop_deseason.plot(ax=axes[0], color="orange", label="Tropics")
axes[0].set_ylabel("T2m / K")
axes[0].legend()

axes[1].scatter(t_deseason, t_trop_deseason)
res = linregress(t_deseason, t_trop_deseason)
axes[1].text(0.02, 0.9, f"r={res.rvalue:.2f}", transform=axes[1].transAxes)
axes[1].set_xlabel("TWP T2m / K")
axes[1].set_ylabel("Tropics T2m / K")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig("plots/twp_tropics_t2m.png", dpi=300, bbox_inches="tight")


# %% calculate lagged correlations
max_lag = 12
lags = np.arange(-max_lag, max_lag + 1)
correlations = {}
for lag in lags:
    temp_lagged = t_deseason.shift(time=lag).dropna("time")
    res = linregress(temp_lagged, t_trop_deseason.sel(time=temp_lagged.time))
    correlations[lag] = res.rvalue

# %% plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(correlations.keys()), list(correlations.values()))
ax.set_xlabel("Lag of TWP to Tropics T2m / months")
ax.set_ylabel("Correlation coefficient")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/lagged_correlation_twp_tropics.png", dpi=300, bbox_inches="tight")

# %%
