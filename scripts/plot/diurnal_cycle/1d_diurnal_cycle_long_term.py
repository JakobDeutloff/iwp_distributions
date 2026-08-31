# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.helper_functions import (
    nan_detrend,
    deseason,
    regress_hist_temp_1d,
)
from scipy.signal import detrend
from scipy.stats import linregress

# %% load ccic data
color = {"all": "black", "sea": "blue", "land": "green"}
names = ["all", "sea", "land"]
dims = {"ccic": "iwp", "gpm": "bt", "era5": "iwp"}
hists_ccic = {}
hists_gpm = {}
hists_era5 = {}
for name in names:
    hists_ccic[name] = xr.open_dataset(
        f"/work/bu1562/m301049/diurnal_cycle_dists/ccic_2d_monthly_{name}.nc"
    )
    hists_gpm[name] = xr.open_dataset(
        f"/work/bu1562/m301049/diurnal_cycle_dists/gpm_2d_monthly_{name}.nc"
    )
for name in ["all", "sea"]:
    hists_era5[name] = xr.open_dataset(
        f"/work/bu1562/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_{name}.nc"
    )
hists_era5["land"] = hists_era5["all"] - hists_era5["sea"]


# %% calculate cloud fraction
cf_ccic = {}
cf_gpm = {}
cf_era5 = {}
for name in names:
    cf_ccic[name] = (
        hists_ccic[name]["hist"].sel(iwp=slice(1, None)).sum("iwp")
        / hists_ccic[name]["size"]
    )
    cf_gpm[name] = (
        hists_gpm[name]["hist"].sel(bt=slice(None, 231)).sum("bt")
        / hists_gpm[name]["size"]
    )
    cf_era5[name] = (
        hists_era5[name]["hist"].sel(iwp=slice(0.37, None)).sum("iwp")
        / hists_era5[name]["size"]
    )

# %% normalise cloud fraction
cf_ccic_norm = {}
cf_gpm_norm = {}
cf_era5_norm = {}
for name in names:
    cf_ccic_norm[name] = cf_ccic[name] / cf_ccic[name].sum("local_time")
    cf_gpm_norm[name] = cf_gpm[name] / cf_gpm[name].sum("local_time")
    cf_era5_norm[name] = cf_era5[name] / cf_era5[name].sum("local_time")

# %% load era5 surface temp
temps = {}
temps["all"] = xr.open_dataset("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").t2m
temps["sea"] = xr.open_dataset(
    "/work/bu1562/m301049/era5/monthly/t2m_tropics_sea.nc"
).t2m
temps["land"] = xr.open_dataset(
    "/work/bu1562/m301049/era5/monthly/t2m_tropics_land.nc"
).t2m


# %%
def regress_hist_temp_1d_trend(cf, temp):

    slopes = xr.zeros_like(cf.isel(time=0))
    p_values = xr.zeros_like(cf.isel(time=0))
    slope_temp, _, _, _, _ = linregress(
        np.arange(len(temp.sel(time=cf.time).values)), temp.sel(time=cf.time).values
    )
    for i in cf.local_time:
        cf_vals = cf.sel({"local_time": i})
        cf_vals = cf_vals.where(np.isfinite(cf_vals), drop=True)
        slope_freq, _, _, p_value, _ = linregress(
            np.arange(len(cf_vals.values)), cf_vals.values
        )
        slopes.loc[{"local_time": i}] = slope_freq / slope_temp
        p_values.loc[{"local_time": i}] = p_value

    slopes_perc = slopes * 100 / cf.mean("time")  # convert to % / K
    return slopes_perc, p_values


slopes_linear_ccic = {}
slopes_linear_gpm = {}
slopes_linear_era5 = {}
p_values_linear_ccic = {}
p_values_linear_gpm = {}
p_values_linear_era5 = {}
for name in names:
    slopes_linear_ccic[name], p_values_linear_ccic[name] = regress_hist_temp_1d_trend(
        cf_ccic_norm[name].fillna(0), temps[name]
    )
    slopes_linear_gpm[name], p_values_linear_gpm[name] = regress_hist_temp_1d_trend(
        cf_gpm_norm[name].fillna(0), temps[name]
    )
    slopes_linear_era5[name], p_values_linear_era5[name] = regress_hist_temp_1d_trend(
        cf_era5_norm[name].fillna(0), temps[name]
    )


# %% plot change of diurnal cycle from trend analysis
def plot_change_diurnal_cycle(slopes, p_values):
    fig, axes = plt.subplots(
        2, 1, figsize=(5, 3.5), height_ratios=[1, 0.2], sharex=True
    )
    axes[0].axhline(0, color="black", linewidth=0.5)
    for name in names:
        axes[0].plot(
            slopes[name].local_time, slopes[name], color=color[name], label=name
        )
        axes[1].plot(
            p_values[name].local_time, p_values[name], color=color[name], alpha=0.5
        )
    for ax in axes:
        ax.set_xlim([0, 24])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([6, 12, 18])

    axes[1].set_xlabel("Local Time / h")
    axes[0].set_ylabel(
        r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
    )
    axes[0].legend()
    axes[1].set_ylim([-0.01, 0.1])
    axes[1].set_ylabel("p-value")

    fig.tight_layout()

    return fig


fig_ccic_change = plot_change_diurnal_cycle(slopes_linear_ccic, p_values_linear_ccic)
fig_gpm_change = plot_change_diurnal_cycle(slopes_linear_gpm, p_values_linear_gpm)
fig_era5_change = plot_change_diurnal_cycle(slopes_linear_era5, p_values_linear_era5)

fig_era5_change.savefig(
    "plots/diurnal_cycle/long_term/era5_1d_change_trend.pdf", bbox_inches="tight"
)

# %% plot change of diurnal cycle from linear trend
offset = {
    "all": -17,
    "sea": -18,
    "land": -19,
}
line_labels = {"all": "All", "sea": "Ocean", "land": "Land"}


def plot_linear_trend(slope, p_value, ax):
    ax.axhline(0, color="black", linewidth=0.5)
    for name in names:
        ax.plot(
            slope[name].local_time,
            slope[name],
            color=color[name],
            label=line_labels[name],
        )
        valid = np.where(p_value[name] < 0.05, offset[name], np.nan)
        ax.plot(p_value[name].local_time, valid, color=color[name], linewidth=4)

    ax.set_xlim([0, 24])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_ylim([-20, 20])
    ax.set_xlabel("Local Time / h")
    ax.set_yticks([-10, 0, 10])


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig_linear_ccic = plot_linear_trend(
    slopes_linear_ccic, p_values_linear_ccic, ax=axes[0]
)
fig_linear_gpm = plot_linear_trend(slopes_linear_gpm, p_values_linear_gpm, ax=axes[1])
axes[0].set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
axes[0].set_title("$I$")
axes[1].set_title("$T_{\mathrm{b}}$")

handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color="grey", linewidth=4))
labels.append("p < 0.05")
fig.legend(handles, labels, bbox_to_anchor=(0.7, -0.05), ncol=4, frameon=False)
fig.savefig("plots/diurnal_cycle/long_term/linear_trend_1d.pdf", bbox_inches="tight")

# %%
fig, ax = plt.subplots()
(cf_ccic_norm["land"] - cf_ccic_norm["land"].mean("time")).plot.pcolormesh(
    "local_time",
    "time",
    ax=ax,
    add_colorbar=True,
    cmap="RdBu_r",
    vmin=-0.003,
    vmax=0.003,
)

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

for i, region in enumerate(["all", "sea"]):
    (
        hists_ccic[region]["hist"].sum("iwp")
        - hists_ccic[region]["hist"].sum("iwp").mean("local_time")
    ).plot.pcolormesh(
        "local_time",
        "time",
        ax=axes[0, i],
        add_colorbar=False,
        cmap="RdBu_r",
        vmin=-1e7,
        vmax=1e7,
    )
    (cf_ccic_norm[region] - cf_ccic_norm[region].mean("time")).plot.pcolormesh(
        "local_time",
        "time",
        ax=axes[1, i],
        add_colorbar=False,
        cmap="RdBu_r",
        vmin=-0.003,
        vmax=0.003,
    )


# %%
