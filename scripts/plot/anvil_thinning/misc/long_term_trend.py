# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, interpolate_bins, load_histograms

# %% initialize containers
bins = np.logspace(-3, 2, 254)[::4]
colors, line_labels, linestyles = definitions()
hists = load_histograms()
hists.pop("dardar")
hists.pop("two_c_ice")

hists["ccic"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
).sum("local_time")


# %% normalise hists
hists_normalized = {}
for key in hists.keys():
    hists_normalized[key] = hists[key]["hist"] / hists[key]["size"]

# %% load era5 surface temp
t_mean = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %% plot all hists
plot_hists(hists_normalized["ccic"], t_mean, bins)

# %% detrend and deseasonalize monthly values
hists_deseason = {}

# temperature
t_detrend = xr.DataArray(detrend(t_mean), coords=t_mean.coords, dims=t_mean.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)

for key in hists_normalized.keys():
    hists_detrend = nan_detrend(hists_normalized[key])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[key] = hists_deseason_ds


# %%regression non-deseasonalized data
slopes = {}
errors = {}
p_vals = {}
for key in hists_normalized.keys():
    slopes_ds = []
    err_ds = []
    p_vals_ds = []
    hist_vals = hists_normalized[key].where(hists_normalized[key].notnull(), drop=True)
    temp = t_mean.sel(time=hist_vals.time)
    for i in range(hists[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
        p_vals_ds.append(res.pvalue)
    slopes[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_normalized[key].iwp},
        dims=["iwp"],
    )
    errors[key] = xr.DataArray(
        err_ds,
        coords={"iwp": hists_normalized[key].iwp},
        dims=["iwp"],
    )
    p_vals[key] = xr.DataArray(
        p_vals_ds,
        coords={"iwp": hists_normalized[key].iwp},
        dims=["iwp"],
    )

# %% regression deseasonalized data
slopes_des = {}
for key in hists_deseason.keys():
    slopes_ds = []
    hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
    temp = t_deseason.sel(time=hist_vals.time)
    for i in range(hists[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
    slopes_des[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )

# %% get just linear trend
slopes_trend = {}
for key in hists_normalized.keys():
    hist_vals = hists_normalized[key].where(hists_normalized[key].notnull(), drop=True)
    temp = t_mean.sel(time=hist_vals.time)
    slopes_ds = []
    for i in range(hists[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i)
        lin_trend_hist = hist_row.polyfit("time", 1)
        lin_trend_temp = temp.polyfit("time", 1)
        slope = lin_trend_hist.polyfit_coefficients.sel(
            degree=1
        ) / lin_trend_temp.polyfit_coefficients.sel(degree=1)
        slopes_ds.append(slope)
    slopes_trend[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_normalized[key].iwp},
        dims=["iwp"],
    )

# %% plot slopes
fig, ax = plt.subplots(figsize=(6, 4))
ax.axhline(0, color="k", linewidth=0.5)

for key in hists_normalized.keys():
    ax.plot(
        slopes_des[key].iwp,
        slopes_des[key],
        color=colors[key],
        label=line_labels[key],
        linestyle="-",
    )
    ax.plot(
        slopes[key].iwp,
        slopes[key],
        label=None,
        color=colors[key],
        linestyle="--",
    )
    ax.plot(
        slopes_trend[key].iwp,
        slopes_trend[key],
        color=colors[key],
        label=None,
        linestyle=":",
    )

handles, labels = ax.get_legend_handles_labels()
handles = handles + [
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
    plt.Line2D([0], [0], color="grey", linestyle=":"),
]
labels = labels + ["Deseasonalized and Detrended", "Simple Regression", "Linear Trend"]
ax.legend(handles, labels, frameon=False, bbox_to_anchor=(0.9, -0.2), ncols=2)

ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
ax.set_ylabel(r"$dP(I)/dT$ / K$^{-1}$")
ax.set_ylim([-0.00091, 0.0014])
ax.set_xlim([0.002, 40])
fig.savefig(
    "plots/anvil_thinning/talk/feedback_monthly_trend_orig_all_trend.pdf",
    bbox_inches="tight",
)

# %% plot timeseries
fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

axes[0].plot(
    hists_normalized["ccic"].time,
    t_mean.sel(time=hists_normalized["ccic"].time, method="nearest"),
    color="k",
    label="ERA5",
)
res = linregress(
    np.arange(len(hists_normalized["ccic"].time)),
    t_mean.sel(time=hists_normalized["ccic"].time, method="nearest").values,
)
trend_line = res.intercept + res.slope * np.arange(len(hists_normalized["ccic"].time))
axes[0].plot(
    hists_normalized["ccic"].time, trend_line, color="r", linestyle="--", label="Trend"
)
axes[1].plot(
    hists_normalized["ccic"].time,
    hists_normalized["ccic"].sel(iwp=10 ** (-1), method="nearest"),
    color="k",
)
res = linregress(
    np.arange(len(hists_normalized["ccic"].time)),
    hists_normalized["ccic"].sel(iwp=10 ** (-1), method="nearest").values,
)
trend_line = res.intercept + res.slope * np.arange(len(hists_normalized["ccic"].time))
axes[1].plot(hists_normalized["ccic"].time, trend_line, color="b", linestyle="--")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("T / K")
axes[1].set_ylabel("P(I=0.1)")

fig.savefig("plots/anvil_thinning/talk/ts_original_trend.pdf", bbox_inches="tight")


# %% calculate linear trend of ccic histograms
linear_trend = {}
for key in hists_normalized.keys():
    hist_vals = hists_normalized[key].where(hists_normalized[key].notnull(), drop=True)
    trend_ds = []
    for i in range(hists[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i)
        lin_trend_hist = hist_row.polyfit("time", 1)
        trend_ds.append(lin_trend_hist.polyfit_coefficients.sel(degree=1).values)
    linear_trend[key] = xr.DataArray(
        trend_ds,
        coords={"iwp": hists_normalized[key].iwp},
        dims=["iwp"],
    )


# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    slopes_trend["ccic"].iwp,
    slopes_trend["ccic"] * 100 / hists_normalized["ccic"].mean("time"),
    color=colors["ccic"],
    label=line_labels["ccic"],
)
ax.plot(
    slopes_trend["spare_ice"].iwp,
    slopes_trend["spare_ice"] * 100 / hists_normalized["spare_ice"].mean("time"),
    color=colors["spare_ice"],
    linestyle="--",
    label=line_labels["spare_ice"],
)
ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
ax.set_ylabel(r"$\dfrac{dP(I)}{P(I) dT}$ / % K$^{-1}$")
ax.legend(frameon=False)
ax.set_ylim([-13, 13])
ax.axhline(0, color="k", linewidth=0.5)
fig.savefig("plots/anvil_thinning/talk/change_frequency.pdf", bbox_inches="tight")


# %% mean IWP from histograms
mean_iwp = {}
for key in hists_normalized.keys():
    mean_iwp[key] = (hists_normalized[key] * hists_normalized[key].iwp).sum("iwp")

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mean_iwp["ccic"].time, mean_iwp["ccic"], color=colors["ccic"], label=line_labels["ccic"])
ax.plot(mean_iwp["spare_ice"].time, mean_iwp["spare_ice"], color=colors["spare_ice"], label=line_labels["spare_ice"])
ax.set_xlabel("Time")
ax.set_ylabel("Mean IWP / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig("plots/anvil_thinning/talk/trend_ccic_mean_iwp.pdf", bbox_inches="tight")

# %%
