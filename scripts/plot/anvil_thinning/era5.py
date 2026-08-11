# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import definitions
from src.helper_functions import nan_detrend, load_histograms, load_slopes

# %% initialize containers
bins = np.logspace(-3, 2, 254)[::4]
colors, line_labels, linestyles = definitions()
hists = load_histograms()
slopes, _, _ = load_slopes()

# add era5
hists["era5"] = xr.open_dataset(
    "/work/bu1562/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_all_weighted.nc"
)
hists["era5"] = hists["era5"].sum("local_time")
hists['era5'] = hists['era5']['hist'] / hists['era5']['size']
# select only ccic and era5
hists = {
    "ccic": hists["ccic"],
    'spare_ice': hists["spare_ice"],
    "era5": hists["era5"],
}

# %% load era5 surface temp
t_mean = xr.open_dataset("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").t2m

# %% detrend and deseasonalize monthly values
hists_deseason = {}

# temperature
t_detrend = xr.DataArray(detrend(t_mean), coords=t_mean.coords, dims=t_mean.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)

for key in hists.keys():
    hists_detrend = nan_detrend(hists[key])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[key] = hists_deseason_ds

# %%regression
slopes_monthly = {}
error_montly = {}
p_vals_monthly = {}
for key in hists_deseason.keys():
    slopes_ds = []
    err_ds = []
    p_vals_ds = []
    hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
    temp = t_deseason.sel(time=hist_vals.time)
    for i in range(hists_deseason[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
        p_vals_ds.append(res.pvalue)
    slopes_monthly[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    error_montly[key] = xr.DataArray(
        err_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    p_vals_monthly[key] = xr.DataArray(
        p_vals_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )

# %% calculate long-term trend
slopes_trend = {}
p_vals_trend = {}
lin_trend_temp = linregress(np.arange(t_mean.size), t_mean.values).slope
for key in hists.keys():
    hist_vals = hists[key].where(hists[key].notnull(), drop=True)
    temp = t_mean.sel(time=hist_vals.time)
    slopes_ds = []
    p_vals_ds = []
    for i in range(hists[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i)
        lin_trend_hist = linregress(np.arange(hist_row.size), hist_row.values)
        slope = lin_trend_hist.slope / lin_trend_temp
        p_val = lin_trend_hist.pvalue
        slopes_ds.append(slope)
        p_vals_ds.append(p_val)
    slopes_trend[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists[key].iwp},
        dims=["iwp"],
    )
    p_vals_trend[key] = xr.DataArray(
        p_vals_ds,
        coords={"iwp": hists[key].iwp},
        dims=["iwp"],
    )

# %% plot all distributions and cre for 2016
fig, ax = plt.subplots(figsize=(8, 6))

for key in hists.keys():
    ax.plot(
        hists[key]["iwp"],
        hists[key].mean('time'),
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

ax.set_xscale("log")
ax.set_xlim([1e-3, 2e1])
ax.set_ylim(0, 0.016)


ax.spines[["top", "right"]].set_visible(False)

ax.legend(frameon=False)
ax.set_ylabel(r"$P(I)$")
ax.set_yticks([0, 0.006, 0.012])
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
fig.savefig('plots/anvil_thinning/era5/era5_dist.png', dpi=300, bbox_inches='tight')

# %% plot slopes and p-value
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])

for key in hists.keys():
    axes[0].plot(
        slopes_monthly[key].iwp,
        slopes_monthly[key],
        label=line_labels[key] + ' variability',
        color=colors[key],
    )
    axes[0].plot(
        slopes_trend[key].iwp,
        slopes_trend[key],
        label=line_labels[key] + " trend",
        color=colors[key],
        linestyle="--",
    )
    axes[1].plot(
        p_vals_monthly[key].iwp,
        p_vals_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )

axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xscale("log")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-3, 2e1)

axes[0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[1].set_ylabel("p-value")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].set_yticks([-0.001, -0.0005, 0, 0.0003])
axes[0].set_ylim(-0.002, 0.0007)
axes[1].set_yticks([0.05, 0.5, 1])
axes[1].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=2, bbox_to_anchor=(0.7, 0))
fig.savefig('plots/anvil_thinning/era5/era5_slopes.png', dpi=300, bbox_inches='tight')

# %% calculate scaling for ERA5
area_fraction_era5 = (
    hists["era5"].mean("time")[::-1]
    .cumsum("iwp")
    .values
)
area_fraction_ccic = (
    hists["ccic"].mean('time')[::-1]
    .cumsum("iwp")
    .values
)
iwp_era5_scaled = np.interp(
    area_fraction_era5, area_fraction_ccic, hists["ccic"]["iwp"][::-1]
)

slopes_monthly["era5"]["iwp"] = iwp_era5_scaled[::-1]
slopes_trend["era5"]["iwp"] = iwp_era5_scaled[::-1]
hists["era5"]["iwp"] = iwp_era5_scaled[::-1]
p_vals_monthly["era5"]["iwp"] = iwp_era5_scaled[::-1]
p_vals_trend["era5"]["iwp"] = iwp_era5_scaled[::-1]
hists["era5"]["iwp"] = iwp_era5_scaled[::-1]


# %% test scaling
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(hists["ccic"]["iwp"][::-1], area_fraction_ccic, color="k", label="CCIC")
ax.plot(hists["era5"]["iwp"][::-1], area_fraction_era5, color="r", label="ERA5")
ax.set_xscale("log")

# %% plot all distributions and cre for 2016
fig, ax = plt.subplots(figsize=(8, 6))

for key in hists.keys():
    ax.plot(
        hists[key]["iwp"],
        hists[key].mean('time'),
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

ax.set_xscale("log")
ax.set_xlim([1e-3, 2e1])
ax.set_ylim(0, 0.016)


ax.spines[["top", "right"]].set_visible(False)

ax.legend(frameon=False)
ax.set_ylabel(r"$P(I)$")
ax.set_yticks([0, 0.006, 0.012])
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
fig.savefig('plots/anvil_thinning/era5/era5_dist_scaled.png', dpi=300, bbox_inches='tight')

# %% plot slopes and p-value
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='row', height_ratios=[3, 1])

for key in hists.keys():
    axes[0, 0].plot(
        slopes_monthly[key].iwp,
        slopes_monthly[key],
        label=line_labels[key], 
        color=colors[key],
    )
    axes[0, 1].plot(
        slopes_trend[key].iwp,
        slopes_trend[key],
        label=line_labels[key], 
        color=colors[key],
    )
    axes[1, 0].plot(
        p_vals_monthly[key].iwp,
        p_vals_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )

    axes[1, 1].plot(
        p_vals_trend[key].iwp,
        p_vals_trend[key],
        label=line_labels[key],
        color=colors[key],
    )

for ax in axes[0, :]:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xscale("log")

for ax in axes[1, :]:
    ax.set_yticks([0.05, 0.5, 1])
    ax.axhline(0.05, color="k", linewidth=0.5)

for ax in axes.flatten():
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-3, 2e1)

axes[0, 0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[1, 0].set_ylabel("p-value")
axes[1, 0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[1, 1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0, 0].set_yticks([-0.001, -0.0005, 0, 0.0003])
axes[0, 0].set_ylim(-0.0019, 0.0007)
axes[1, 0].set_yticks([0.05, 0.5, 1])
axes[1, 0].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, bbox_to_anchor=(0.7, 0))

# add letters
for i, ax in enumerate(axes.flatten()):
    ax.text(0.03, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.savefig('plots/anvil_thinning/publication/era5_slopes_scaled.pdf', bbox_inches='tight')

# %% test era5 cf 
ds_cf = xr.open_dataset("/work/bu1562/m301049/era5/hourly/cf_2016.nc").isel(valid_time=slice(0, 25)).load()

# %% plot hist of cf 
fig, ax = plt.subplots(figsize=(8, 6))
cf_hist, cf_bins = np.histogram(ds_cf['hcc'].values.flatten(), bins=np.arange(0, 1.05, 0.05), density=True)
ax.stairs(cf_hist, cf_bins, label='ERA5 Cloud Fraction', color='blue')
ax.set_xlabel('Cloud Fraction')
ax.set_ylabel('Probability Density')
ax.set_title('Histogram of Cloud Fraction (ERA5, 2016)')
# %%
