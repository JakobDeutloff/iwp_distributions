# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.helper_functions import (
    nan_detrend,
    read_ccic_dc,
    resample_histograms,
    deseason,
    regress_hist_temp_1d,
)
from scipy.signal import detrend
from scipy.stats import linregress

# %% load ccic data
color = {"all": "black", "sea": "blue", "land": "green"}
names = ["all", "sea", "land"]
hists = {}
hists["sea"] = read_ccic_dc("ccic_cpcir_daily_cycle_distribution_sea_")
hists["all"] = read_ccic_dc("ccic_cpcir_daily_cycle_distribution_")
hists["land"] = hists["all"] - hists["sea"]

# %% resample histograms to monthly
hists_monthly = {}
for name in names:
    hists_monthly[name] = resample_histograms(hists[name])

# %% cut
for name in names:
    hists_monthly[name] = hists_monthly[name].sel(time=slice(None, '2023-01'))

# %% load era5 surface temp
temps = {}
temps["all"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics.nc"
).t2m
temps["sea"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc"
).t2m
temps["land"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_land.nc"
).t2m

# %% detrend and deseasonalize
temps_deseason = {}
for name in names:
    temp_detrend = xr.DataArray(
        detrend(temps[name]), coords=temps[name].coords, dims=temps[name].dims
    )
    temps_deseason[name] = deseason(temp_detrend)
hists_deseason = {}
for name in names:
    hist_detrend = nan_detrend(hists_monthly[name], dim="local_time")
    hists_deseason[name] = deseason(hist_detrend)

# %% regression
slopes = {}
err = {}
for name in names:
    slopes[name], err[name] = regress_hist_temp_1d(
        hists_deseason[name], temps_deseason[name]
    )

# %% plot mean histograms
mean_hists = {}
fig, ax = plt.subplots(figsize=(8, 5))
for name in names:
    mean_hists[name] = (
        hists[name]["hist"].sum("time")
        / hists[name]["hist"].sum(["time", "local_time"])
    )
    ax.stairs(
        mean_hists[name],
        np.arange(0, 25, 1),
        color=color[name],
        label=name,
        linewidth=2,
    )

ax.set_ylim([0.02, 0.075])
ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)



# %% plot 
fig, ax = plt.subplots(figsize=(8, 5))
for name in names:
    slope_perc = slopes[name]*100/mean_hists[name]
    ax.plot(slopes[name].local_time, slope_perc, color=color[name], label=name)
    ax.fill_between(
        slopes[name].local_time,
        slope_perc - err[name]*100/mean_hists[name],
        slope_perc + err[name]*100/mean_hists[name],
        color=color[name],
        alpha=0.3,
    )

ax.axhline(0, color="black", linestyle="--")
ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("dP/dT / % K$^{-1}$")
ax.legend()
ax.spines[["top", "right"]].set_visible(False) 


# %%
