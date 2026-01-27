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


# %% load ccic data
color = {"all": "black", "sea": "blue", "land": "green"}
names = ["all", "sea", "land"]
dims = {"ccic": "iwp", "gpm": "bt"}
hists_ccic = {}
hists_gpm = {}
for name in names:
    hists_ccic[name] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_{name}.nc"
    )
    hists_gpm[name] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_{name}.nc"
    )

# %% calculate cloud fraction
cf_ccic = {}
cf_gpm = {}
for name in names:
    cf_ccic[name] = hists_ccic[name]["hist"].sel(iwp=slice(1, None)).sum(
        "iwp"
    ) / hists_ccic[name]["hist"].sum(["iwp", "local_time"])
    cf_gpm[name] = hists_gpm[name]["hist"].sel(bt=slice(None, 232)).sum(
        "bt"
    ) / hists_gpm[name]["hist"].sum(['bt', "local_time"])

# %% normalise  cloud fractions
cf_ccic_norm = {}
cf_gpm_norm = {}
for name in names:
    cf_ccic_norm[name] = cf_ccic[name] / cf_ccic[name].sum("local_time")
    cf_gpm_norm[name] = cf_gpm[name] / cf_gpm[name].sum("local_time")


# %% load era5 surface temp
temps = {}
temps["all"] = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m
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
cf_ccic_deseason = {}
cf_gpm_deseason = {}
for name in names:
    hist_detrend = nan_detrend(cf_ccic_norm[name], dim="local_time")
    cf_ccic_deseason[name] = deseason(hist_detrend)
    hist_detrend = nan_detrend(cf_gpm_norm[name], dim="local_time")
    cf_gpm_deseason[name] = deseason(hist_detrend)

# %% regression
slopes_ccic = {}
slopes_gpm = {}
err_ccic = {}
err_gpm = {}
for name in names:
    slopes_ccic[name], err_ccic[name] = regress_hist_temp_1d(
        cf_ccic_deseason[name], temps_deseason[name], cf_ccic_norm[name]
    )
    slopes_gpm[name], err_gpm[name] = regress_hist_temp_1d(
        cf_gpm_deseason[name], temps_deseason[name], cf_gpm_norm[name]
    )

# %% load icon
runs = ["jed0011", "jed0022", "jed0033"]
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists_icon = {}
hists_raw = {}
slopes_icon = {}
for run in runs:
    hists_raw[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/publication/distributions/{run}_deep_clouds_daily_cycle.nc"
    )
    hists_icon[run] = (hists_raw[run].sum("day") / hists_raw[run].sum())[
        "__xarray_dataarray_variable__"
    ].values


for run in runs[1:]:
    slopes_icon[run] = (
        (hists_icon[run] - hists_icon["jed0011"])
        * 100
        / temp_delta[run]
        / hists_icon["jed0011"]
    )


# %% plot mean histograms
dims = {"ccic": "iwp", "gpm": "bt"}


def plot_mean_histograms(cf):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name in names:
        ax.plot(
            cf[name].local_time,
            cf[name].mean("time"),
            color=color[name],
            label=name,
            linewidth=2,
        )

    ax.set_xlim([0, 24])
    ax.set_xlabel("Local Time / h")
    ax.set_ylabel("$f$")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_yticks([0.001, 0.0015, 0.002, 0.0025])
    fig.tight_layout()

    return fig


fig_ccic = plot_mean_histograms(cf_ccic)
fig_ccic.savefig("plots/diurnal_cycle/publication/mean_ccic_diurnal_cycle_land_sea.pdf")
fig_gpm = plot_mean_histograms(cf_gpm)
fig_gpm.savefig(
    "plots/diurnal_cycle/mean_gpm_diurnal_cycle_land_sea.pdf",
)

# %% calculate total cf 
total_cf_ccic = {}
total_cf_gpm = {}
for name in names:
    total_cf_ccic[name] = cf_ccic[name].sum("local_time").mean('time')
    print(f'{name} total ccic cf: {total_cf_ccic[name].values}')
    total_cf_gpm[name] = cf_gpm[name].sum("local_time").mean('time')
    print(f'{name} total gpm cf: {total_cf_gpm[name].values}')


# %% plot change of diurnal cycle
def plot_change_diurnal_cycle(slopes, err):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axhline(0, color="black", linewidth=0.5)
    for name in names:
        ax.plot(slopes[name].local_time, slopes[name], color=color[name], label=name)
        ax.fill_between(
            slopes[name].local_time,
            slopes[name] - err[name],
            slopes[name] + err[name],
            color=color[name],
            alpha=0.3,
        )

    ax.set_xlim([0, 24])
    ax.set_xlabel("Local Time / h")
    ax.set_ylabel(r"$\dfrac{\mathrm{d}f}{f~\mathrm{d}T}$ / % K$^{-1}$")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    fig.tight_layout()

    return fig


fig_ccic_change = plot_change_diurnal_cycle(slopes_ccic, err_ccic)
fig_gpm_change = plot_change_diurnal_cycle(slopes_gpm, err_gpm)

# %% make plot for paper
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax in axes:
    ax.axhline(0, color="black", linewidth=0.5)

for name in ["land", "sea"]:
    axes[1].plot(
        slopes_ccic[name].local_time,
        slopes_ccic[name],
        color=color[name],
        label=f"$I$ {name}",
    )
    axes[1].fill_between(
        slopes_ccic[name].local_time,
        slopes_ccic[name] - err_ccic[name],
        slopes_ccic[name] + err_ccic[name],
        color=color[name],
        alpha=0.3,
    )
    axes[1].plot(
        slopes_gpm[name].local_time,
        slopes_gpm[name],
        color=color[name],
        linestyle="--",
        label=rf"$T_{{\mathrm{{b}}}} ~ \mathrm{{{name}}}$",
    )
    axes[1].fill_between(
        slopes_gpm[name].local_time,
        slopes_gpm[name] - err_gpm[name],
        slopes_gpm[name] + err_gpm[name],
        color=color[name],
        alpha=0.3,
    )
axes[0].plot(
    slopes_ccic["all"].local_time,
    slopes_ccic["all"],
    color="black",
    label=f"$I$ all",
    linestyle="-",
)
axes[0].fill_between(
    slopes_ccic["all"].local_time,
    slopes_ccic["all"] - err_ccic["all"],
    slopes_ccic["all"] + err_ccic["all"],
    color="black",
    alpha=0.3,
)
axes[0].plot(
    slopes_gpm["all"].local_time,
    slopes_gpm["all"],
    color="k",
    label=r"$T_{\mathrm{b}}$ all",
    linestyle="--",
)
axes[0].fill_between(
    slopes_gpm["all"].local_time,
    slopes_gpm["all"] - err_gpm["all"],
    slopes_gpm["all"] + err_gpm["all"],
    color="k",
    alpha=0.3,
)
axes[0].plot(
    slopes_ccic["all"].local_time,
    slopes_icon["jed0022"],
    color="red",
    label="ICON +4K",
)

for ax in axes:
    ax.set_xlim([0, 24])
    ax.set_xlabel("Local Time / h")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_yticks([-4, 0, 4])

# add letters 
for ax, letter in zip(axes, ["a", "b"]):
    ax.text(
        0.05,
        1,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
    )


axes[0].set_ylabel(r"$\dfrac{\mathrm{d}f}{f~\mathrm{d}T}$ / % K$^{-1}$")
fig.tight_layout()

fig.savefig("plots/diurnal_cycle/publication/diurnal_cycle_change_land_sea_paper.pdf")

# %%
