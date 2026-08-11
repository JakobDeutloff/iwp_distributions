# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, interpolate_bins, load_histograms
import pickle


# %% initialize containers
bins = np.logspace(-3, 2, 254)[::4]
path = "/work/bm1183/m301049/diurnal_cycle_dists/"
hists = {
    "all": xr.open_dataset(path + "ccic_2d_monthly_all.nc").sum('local_time'),
    "sea": xr.open_dataset(path + "ccic_2d_monthly_sea.nc").sum('local_time'),
    "land": xr.open_dataset(path + "ccic_2d_monthly_land.nc").sum('local_time'),
}
temp = {
    "all": xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m,
    "sea": xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc").t2m,
    "land": xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics_land.nc").t2m
}

#%% normalise hists
hists_normalized = {}
for key in hists.keys():
    hists_normalized[key] = hists[key]["hist"] / hists[key]["size"]

# %% detrend and deseasonalize monthly values
hists_deseason = {}
temp_deseason = {}

for key in hists_normalized.keys():
    temp_detrend = xr.DataArray(detrend(temp[key]), coords=temp[key].coords, dims=temp[key].dims)
    temp_deseason[key] = temp_detrend.groupby("time.month") - temp_detrend.groupby("time.month").mean(
        "time"
    )
    hists_detrend = nan_detrend(hists_normalized[key])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[key] = hists_deseason_ds

# %%regression
slopes = {}
error = {}
p_vals = {}
for key in hists_deseason.keys():
    slopes_ds = []
    err_ds = []
    p_vals_ds = []
    hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
    temp = temp_deseason[key].sel(time=hist_vals.time)
    for i in range(hists_deseason[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
        p_vals_ds.append(res.pvalue)
    slopes[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    error[key] = xr.DataArray(
        err_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    p_vals[key] = xr.DataArray(
        p_vals_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )

# %% calculate feedback 
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)
cre["iwp"] = np.log10(cre["iwp"])
cre = cre.interp(
    iwp=np.log10(hists["all"].iwp), method="linear"
).drop_vars("iwp")
cre["iwp"] = hists["all"].iwp
feedback = {}
for key in slopes.keys():
    feedback[key] = slopes[key] * cre["net"].values

# %% calculate combined feedback 
sea_fraction = 0.73
combined_feedback = feedback["sea"] * sea_fraction + feedback["land"] * (1 - sea_fraction)

# %% plot distributions 
line_labels = {
    "all": "All",
    "sea": "Sea",
    "land": "Land",
}
colors = {
    "all": 'k',
    "sea": 'b',
    "land": 'brown',
}
fig, ax = plt.subplots(figsize=(8, 4))
for key in hists.keys():
    ax.plot(
        hists_normalized[key].iwp[1:],
        hists_normalized[key].mean('time')[1:],
        label=line_labels[key],
        color=colors[key],
    )

ax.set_xscale("log")
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
ax.set_ylabel("$P(I)$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([1e-3, 2e1])
ax.set_ylim(0, 0.015)
ax.legend()
fig.savefig("plots/anvil_thinning/land_sea/distributions.png", dpi=300, bbox_inches='tight')


# %% plot slopes 
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})


for key in hists.keys():
    axes[0].plot(
        slopes[key].iwp[1:],
        slopes[key][1:],
        label=line_labels[key],
        color=colors[key],
    )
    axes[1].plot(
        p_vals[key].iwp[1:],
        p_vals[key][1:],
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
axes[0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[0].set_ylim(-0.0006, 0.0004)
axes[1].set_yticks([0.05, 0.5, 1])
axes[1].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, bbox_to_anchor=(0.75, 0))
fig.savefig("plots/anvil_thinning/land_sea/slopes.png", dpi=300, bbox_inches='tight')

# %% plot feedback 
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])

for key in hists.keys():
    axes[0].plot(
        feedback[key].iwp[1:],
        feedback[key][1:],
        label=line_labels[key],
        color=colors[key],
    )
    axes[1].scatter(
        1,
        feedback[key][1:].sum(),
        label=line_labels[key],
        color=colors[key],
    )

axes[1].scatter(
    1,
    combined_feedback[1:].sum(),
    label="Combined",
    color='k',
    marker='x'
)

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)



axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False, loc="upper left")
axes[0].set_yticks([-0.02, 0, 0.02])
axes[1].set_ylabel(r"$\lambda$ / W m$^{-2}$ K$^{-1}$")
axes[1].spines['bottom'].set_visible(False)
axes[1].set_xticks([])

fig.tight_layout()
fig.savefig("plots/anvil_thinning/land_sea/feedback.png", dpi=300, bbox_inches='tight')

# %%
