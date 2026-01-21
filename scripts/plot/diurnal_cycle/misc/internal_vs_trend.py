# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.helper_functions import (
    nan_detrend,
    resample_histograms,
    deseason,
    regress_hist_temp_1d,
    lowpass_filter,
)
from src.plot import definitions
from scipy.signal import detrend
from scipy.stats import linregress

# %%
colors, line_labels, linestyles = definitions()
names = ["trend", "internal", "lowpass-trend", "lowpass-internal"]
color = {
    "trend": "red",
    "internal": "blue",
    "lowpass-trend": "orange",
    "lowpass-internal": "cyan",
}

# %% open histograms
hist_gpm = xr.open_mfdataset("/work/bm1183/m301049/GPM_MERGIR/hists/gpm_2d_hist_all*.nc").load()
hist_gpm = hist_gpm.sel(bt=slice(None, 240)).sum("bt")
hist_gpm_month = resample_histograms(hist_gpm)

# %% load era5 surface temp
temp_trop = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m


# %%  detrend and deseasonalize
def rolling_average(da, window=3):
    return da.rolling(time=window, center=True).mean()


temps = {}
temps["trend"] = rolling_average(deseason(temp_trop))
temps["internal"] = rolling_average(
    deseason(
        xr.DataArray(
            detrend(temp_trop, axis=0), coords=temp_trop.coords, dims=temp_trop.dims
        )
    )
)
temps["lowpass-trend"] = lowpass_filter(temp_trop, cutoff_period_years=2)
temps["lowpass-internal"] = xr.DataArray(
    detrend(temps["lowpass-trend"], axis=0),
    coords=temps["lowpass-trend"].coords,
    dims=temps["lowpass-trend"].dims,
)

hists = {}
hists["trend"] = rolling_average(deseason(hist_gpm_month))
hists["internal"] = rolling_average(
    deseason(nan_detrend(hist_gpm_month, dim="local_time"))
)
hists["lowpass-trend"] = lowpass_filter(hist_gpm_month, cutoff_period_years=2)
hists["lowpass-internal"] = nan_detrend(hists["lowpass-trend"], dim="local_time")


# %% regression
slopes = {}
err = {}

for name in names:
    slopes[name], err[name] = regress_hist_temp_1d(hists[name], temps[name])

# %% plot change in diurnal cycle
fig, axes = plt.subplots(2, 1, figsize=(8, 5))
axes[0].axhline(0, color="k", linewidth=0.5)

for name in names:
    mean_hist = hist_gpm["hist"].sum("time") / hist_gpm["hist"].sum(
        ["time", "local_time"]
    )
    axes[0].plot(
        mean_hist["local_time"],
        slopes[name],
        label=f"{name}",
        color=color[name],
    )
    axes[1].plot(
        mean_hist["local_time"],
        err[name],
        label=f"{name}",
        color=color[name],
    )

axes[0].set_ylabel("dP/dT / % K$^{-1}$")
axes[1].set_xlabel("Local Time / h")
axes[1].set_ylabel("p-value")


for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([0, 23.9])

axes[0].legend()
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/sensitivity/timings.png", dpi=300, bbox_inches="tight")

# %% plot timeseries of histograms at lt=12.5
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for name in names:
    hist_lt = hists[name].sel(local_time=12.5, method="nearest")
    axes[0].plot(
        hist_lt["time"],
        hist_lt - hist_lt.mean(),
        label=f"{name}",
        color=color[name],
    )
    axes[0].set_ylabel("P")
    axes[1].plot(
        temps[name].sel(time=hist_lt.time)["time"],
        temps[name].sel(time=hist_lt.time) - temps[name].sel(time=hist_lt.time).mean(),
        label=f"{name}",
        color=color[name],
    )
    axes[0].legend()
    axes[1].set_ylabel("T / K")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/sensitivity/timeseries_lt12.5.png", dpi=300)

# %% scatterplot of lowpass trends at lt=12.5
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(
    temps["trend"].sel(time=hists["trend"].time),
    hists["trend"].sel(local_time=14.5),
    color=color["trend"],
)

axes[1].scatter(
    temps["internal"].sel(time=hists["internal"].time),
    hists["internal"].sel(local_time=14.5),
    color=color["internal"],
)

# %% 
def calculate_change(start, end, hist, temp):

    hist = hist.sel(time=slice(start, end))
    temp = temp.sel(time=slice(start, end))

    hist_trend = lowpass_filter(hist, cutoff_period_years=2)
    temp_trend = lowpass_filter(temp, cutoff_period_years=2)

    hist_internal = nan_detrend(hist_trend, dim="time")
    temp_internal = xr.DataArray(
        detrend(temp_trend, axis=0),
        coords=temp_trend.coords,
        dims=temp_trend.dims,
    )

    slope_trend, _ = regress_hist_temp_1d(hist_trend, temp_trend)
    slope_internal, _ = regress_hist_temp_1d(hist_internal, temp_internal)

    x = np.arange(temp_trend.time.size)
    res = linregress(x, temp_trend.values)
    trend = res.slope * (x[-1]) - res.slope * (x[0]) 
    difference = temp_internal.max().values - temp_internal.min().values

    return slope_trend, slope_internal, trend, difference


# %% plot cahnge for different start and enddates 
periods = [
    (None, None), 
    ("2001-01", None),
    ("2002-01", None),
    ("2003-01", None),
    ("2004-01", None),
    (None, '2023-01'),
    (None, '2022-01'),
    (None, '2021-01'),
    (None, '2020-01'), 
]

slopes_trend = []
slopes_internal = []
trends = []
differences = []
for start, end in periods:
    slope_trend, slope_internal, trend, difference = calculate_change(
        start, end, hist_gpm_month, temp_trop
    )
    slopes_trend.append(slope_trend)
    slopes_internal.append(slope_internal)
    trends.append(trend)
    differences.append(difference)

temp_trend = lowpass_filter(temp_trop, cutoff_period_years=2).sel(time=hist_gpm_month.time)
temp_internal = xr.DataArray(detrend(temp_trend, axis=0), coords=temp_trend.coords, dims=temp_trend.dims)
# %% plot slopes for different periods
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
#make alpha range from 0.5 to 1.0
alpha = np.linspace(1.0, 0.2, len(periods))

for i, (slope_trend, slope_internal) in enumerate(zip(slopes_trend, slopes_internal)):
    axes[0].plot(
        hist_gpm["local_time"],
        slope_trend,
        label=f"from {periods[i][0] or 'start'} to {periods[i][1] or 'end'}",
        color='k',
        alpha=alpha[i],
    )
    axes[0].plot(
        hist_gpm["local_time"],
        slope_internal,
        color='red',
        alpha=alpha[i],
    )
axes[1].plot(
    temp_trend["time"],
    temp_trend - temp_trend.mean(),
    label="Trend",
    color="black",
)
axes[1].plot(
    hist_gpm_month["time"],
    temp_internal - temp_internal.mean(),
    label="Internal",
    color="red",
)
axes[0].set_ylabel("dP/dT / K$^{-1}$")
axes[0].set_xlabel("Local Time / h")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("T' / K")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/sensitivity/slopes_all_periods.png", dpi=300, bbox_inches="tight")

# %%
print(f"Mean trend in T: {np.mean(trends):.2f} K")
print(f"Mean internal variability in T: {np.mean(differences):.2f} K")

# %%
