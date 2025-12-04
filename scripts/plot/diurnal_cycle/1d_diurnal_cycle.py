# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
from dask.diagnostics import ProgressBar
from src.helper_functions import nan_detrend, resample_histograms, deseason, regress_hist_temp_1d
from src.plot import definitions
from scipy.signal import detrend
from scipy.stats import linregress

# %%
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange", "icon": "green"}
names = ["ccic", "gpm"]
# %% open histograms
hists = {}
# ccic
hist_ccic = xr.open_mfdataset(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d_*.nc"
).load()

# gpm
hist_gpm = xr.open_mfdataset("/work/bm1183/m301049/GPM_MERGIR/hists/gpm_*.nc").load()

# %% cut histograms
hists["ccic"] = hist_ccic.sel(iwp=slice(1e0, None)).sum("iwp")
hists["gpm"] = hist_gpm.sel(bt=slice(None, 237)).sum("bt")

# %%
hists_monthly = {}
for name in names:
    hists_monthly[name] = resample_histograms(hists[name])

# %% load era5 surface temp
temp_trop = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %%  detrend and deseasonalize
temp_detrend = xr.DataArray(detrend(temp_trop), coords=temp_trop.coords, dims=temp_trop.dims)
temp = deseason(temp_trop)
for name in names:
    hist_detrend = nan_detrend(hists_monthly[name], dim="local_time")
    hists_monthly[name] = deseason(hist_detrend)

# %% regression
slopes = {}
err = {}

for name in names:
    slopes[name], err[name] = regress_hist_temp_1d(hists_monthly[name], temp)

# %% load icon
runs = ["jed0011", "jed0022", "jed0033"]
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists_icon = {}
hists_raw = {}
for run in runs:
    hists_raw[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/publication/distributions/{run}_deep_clouds_daily_cycle.nc"
    )
    hists_icon[run] = (hists_raw[run].sum("day") / hists_raw[run].sum())[
        "__xarray_dataarray_variable__"
    ].values


change_icon = {}
for run in runs[1:]:
    change_icon[run] = (hists_icon[run] - hists_icon["jed0011"]) / temp_delta[run]

# %% plot of mean daily cycle
fig, ax = plt.subplots(figsize=(8, 5))
for name in names:
    mean_hist = hists[name]['hist'].sum('time') / hists[name]['hist'].sum(['time', 'local_time'])
    ax.stairs(
        mean_hist,
        np.arange(0, 25, 1),
        label=f"{name}",
        color=color[name],
        linewidth=2,
    )
for run in runs:
    ax.stairs(
        hists_icon[run], np.arange(0, 25, 1), label=line_labels[run], color=colors[run]
    )
ax.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax.set_ylim([0.03, 0.055])
ax.set_yticks([0.03, 0.04, 0.05])
ax.set_xlim([0, 23.9])
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/diurnal_cycle_mean.png", dpi=300, bbox_inches="tight")


# %% plot change in diurnal cycle
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color="k", linewidth=0.5)
mean_ccic = (
    hists['ccic']['hist'].sum("time") / hists['ccic']['hist'].sum(["time", 'local_time'])
)

for name in names:
    mean_hist = hists[name]['hist'].sum('time') / hists[name]['hist'].sum(['time', 'local_time'])
    ax.plot(
        mean_hist["local_time"],
        slopes[name] * 100 / mean_hist,
        label=f"{name}",
        color=color[name],
    )
    ax.fill_between(
        mean_hist["local_time"],
        (slopes[name] - err[name]) * 100 / mean_hist,
        (slopes[name] + err[name]) * 100 / mean_hist,
        color=color[name],
        alpha=0.3,
    )

for run in runs[1:]:
    ax.plot(
        slopes['ccic']["local_time"],
        change_icon[run] * 100 / hists_icon["jed0011"],
        label=line_labels[run],
        color=colors[run],
    )
ax.set_ylabel("dP/dT / % K$^{-1}$")
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([0, 23.9])
ax.legend()
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/diurnal_cycle_change.png", dpi=300, bbox_inches="tight")

# %%
