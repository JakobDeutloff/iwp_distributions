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


# %% load ccic and gpm data
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
    hists_ccic[name] = hists_ccic[name].groupby("time.year").mean("time").rename(year="time")
    hists_gpm[name] = hists_gpm[name].groupby("time.year").mean("time").rename(year="time")
# %% calculate cloud fraction
cf_ccic = {}
cf_gpm = {}
for name in names:
    cf_ccic[name] = hists_ccic[name]["hist"].sel(iwp=slice(1, None)).sum(
        "iwp"
    ) / hists_ccic[name]["hist"].sum(["iwp", "local_time"])
    cf_gpm[name] = hists_gpm[name]["hist"].sel(bt=slice(None, 231)).sum(
        "bt"
    ) / hists_gpm[name]["hist"].sum(["bt", "local_time"])

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
for name in names:
    temps[name] = temps[name].groupby("time.year").mean("time").rename(year="time")

# %% detrend and deseasonalize
temps_deseason = {}
for name in names:
    temp_detrend = xr.DataArray(
        detrend(temps[name]), coords=temps[name].coords, dims=temps[name].dims
    )
    temps_deseason[name] = temp_detrend #deseason(temp_detrend)
cf_ccic_deseason = {}
cf_gpm_deseason = {}
for name in names:
    cf_detrend = nan_detrend(cf_ccic_norm[name], dim="local_time")
    cf_ccic_deseason[name] = cf_detrend #deseason(cf_detrend)
    cf_detrend = nan_detrend(cf_gpm_norm[name], dim="local_time")
    cf_gpm_deseason[name] = cf_detrend #deseason(cf_detrend)

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
        f"/work/bm1183/m301049/icon_hcap_data/publication/distributions/{run}_daily_cycle_hist_2d.nc"
    )
    hists_icon[run] = hists_raw[run]["hist"].sel(iwp=slice(1, None)).sum("iwp") / hists_raw[run]["size"]


for run in runs[1:]:
    slopes_icon[run] = (
        (hists_icon[run] - hists_icon["jed0011"])
        * 100
        / temp_delta[run]
        / hists_icon["jed0011"]
    )

# %% plot diurnal cycle of both
fig, ax = plt.subplots(figsize=(6, 4))

for name in names:
    ax.plot(
        cf_ccic[name].local_time,
        cf_ccic[name].mean("time"),
        color=color[name],
        linestyle="-",
    )

ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("$f_{\mathrm{d}}$")
handles = [
    plt.Line2D([0], [0], color="black", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="green", linestyle="-"),]
labels = ["All", "Ocean", "Land"]

ax.legend(handles, labels, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([6, 12, 18])
ax.set_yticks([0.001, 0.002, 0.003])
#fig.savefig("plots/diurnal_cycle/publication/mean_dc.pdf", bbox_inches="tight")

# %% calculate total cf
total_cf_ccic = {}
total_cf_gpm = {}
for name in names:
    total_cf_ccic[name] = cf_ccic[name].sum("local_time").mean("time")
    print(f"{name} total ccic cf: {total_cf_ccic[name].values}")
    total_cf_gpm[name] = cf_gpm[name].sum("local_time").mean("time")
    print(f"{name} total gpm cf: {total_cf_gpm[name].values}")


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
    ax.set_ylabel(
        r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    fig.tight_layout()

    return fig


fig_ccic_change = plot_change_diurnal_cycle(slopes_ccic, err_ccic)
fig_gpm_change = plot_change_diurnal_cycle(slopes_gpm, err_gpm)
#fig_ccic_change.savefig("plots/diurnal_cycle/talk/ccic_1d_change.pdf", bbox_inches="tight")
#fig_gpm_change.savefig("plots/diurnal_cycle/talk/gpm_1d_change.pdf", bbox_inches="tight")

# %% calculate mean change in f
for name in names:
    mean_change_ccic = (slopes_ccic[name] * cf_ccic[name].mean("time")).mean(
        "local_time"
    )
    print(f"{name} mean ccic change in f: {mean_change_ccic.values}  K^-1")
    mean_change_gpm = (slopes_gpm[name] * cf_gpm[name].mean("time")).mean("local_time")
    print(f"{name} mean gpm change in f: {mean_change_gpm.values}  K^-1")

# %% make plot for paper
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
labels = {
    "all": "All",
    "sea": "Ocean",
    "land": "Land",
}
for ax in axes:
    ax.axhline(0, color="black", linewidth=0.5)

for name in ["land", "sea"]:
    axes[1].plot(
        slopes_ccic[name].local_time,
        slopes_ccic[name],
        color=color[name],
        label=f"{labels[name]}",
    )
    axes[1].fill_between(
        slopes_ccic[name].local_time,
        slopes_ccic[name] - err_ccic[name],
        slopes_ccic[name] + err_ccic[name],
        color=color[name],
        alpha=0.3,
    )
axes[0].plot(
    slopes_ccic["all"].local_time,
    slopes_ccic["all"],
    color="black",
    label=f"$I$ All",
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
    label=r"$T_{\mathrm{b}}$ All",
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
    label="GSRM +4K",
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


axes[0].set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
fig.tight_layout()

fig.savefig("plots/diurnal_cycle/publication/diurnal_cycle_change_land_sea_paper.pdf")

# %% numbers for paper
print(f"ccic: {slopes_ccic['all'].min()}")
print(f"gpm: {slopes_gpm['all'].min()}")

# %% plot non-normalised change in f_d
# detrend and deseasonalize
cf_ccic_deseason_raw = {}
cf_gpm_deseason_raw = {}
for name in names:
    cf_detrend = nan_detrend(cf_ccic[name], dim="local_time")
    cf_ccic_deseason_raw[name] = deseason(cf_detrend)
    cf_detrend = nan_detrend(cf_gpm[name], dim="local_time")
    cf_gpm_deseason_raw[name] = deseason(cf_detrend)

#regression
slopes_ccic_raw = {}
slopes_gpm_raw = {}
err_ccic_raw = {}
err_gpm_raw = {}
for name in names:
    slopes_ccic_raw[name], err_ccic_raw[name] = regress_hist_temp_1d(
        cf_ccic_deseason_raw[name], temps_deseason[name], cf_ccic[name]
    )
    slopes_gpm_raw[name], err_gpm_raw[name] = regress_hist_temp_1d(
        cf_gpm_deseason_raw[name], temps_deseason[name], cf_gpm[name]
    )

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.axhline(0, color="black", linewidth=0.5)
ax.plot(
    slopes_ccic_raw["all"].local_time,
    slopes_ccic_raw["all"],
    color="black",
    label = f"$I$ All",
)
ax.fill_between(
    slopes_ccic_raw["all"].local_time,
    slopes_ccic_raw["all"] - err_ccic_raw["all"],
    slopes_ccic_raw["all"] + err_ccic_raw["all"],
    color="black",
    alpha=0.3,
)
ax.plot(
    slopes_gpm_raw["all"].local_time,
    slopes_gpm_raw["all"],
    color="k",
    linestyle="--",
    label=r"$T_{\mathrm{b}}$ All",
)
ax.fill_between(
    slopes_gpm_raw["all"].local_time,
    slopes_gpm_raw["all"] - err_gpm_raw["all"],
    slopes_gpm_raw["all"] + err_gpm_raw["all"],
    color="k",
    alpha=0.3,
)

ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([6, 12, 18])
ax.legend(frameon=False)
fig.savefig("plots/diurnal_cycle/publication/diurnal_cycle_change_all_nonnormalised.pdf", bbox_inches="tight")

# %% plot for thesis
hist_icon_control = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon_hcap_data/control/production/daily_cycle_hist_2d.nc"
    )
    .coarsen(iwp=4, boundary="trim")
    .sum()
)
cf_icon = (
    hist_icon_control["hist"].sel(iwp=slice(1, None)).sum("iwp")
    / hist_icon_control["size"]
)

color = {
    'land': "#9B6A01",
    'sea': "#3d9ef8",
}
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axes[1].axhline(0, color="black", linewidth=0.7)
# plot diurnal cycle CCIC and ICON 
for name in ["land", "sea"]:
    axes[0].plot(
        cf_ccic[name].local_time,
        cf_ccic[name].mean("time") / cf_ccic[name].mean(),
        color=color[name],
        linestyle="-",
        label=f"CCIC {name}",
    )
axes[0].plot(
    cf_icon.local_time,
    cf_icon.mean("time") / cf_icon.mean(),
    color="#462d7b",
    linestyle="--",
    label="ICON",
)
# plot change of diurnal cycle
for name in ["land", "sea"]:
    axes[1].plot(
        slopes_ccic[name].local_time,
        slopes_ccic[name],
        color=color[name],
        label=f"CCIC {name}",
    )
    axes[1].fill_between(
        slopes_ccic[name].local_time,
        slopes_ccic[name] - err_ccic[name],
        slopes_ccic[name] + err_ccic[name],
        color=color[name],
        alpha=0.3,
    )
axes[1].plot(
    slopes_ccic[name].local_time,
    slopes_icon["jed0022"],
    color="#c1df24",
    label="ICON +4K",
    linestyle="--",
)
axes[1].plot(
    slopes_ccic[name].local_time,
    slopes_icon['jed0033'],
    color="#1f948a",
    label="ICON +2K",
    linestyle="--",
)
axes[0].set_ylabel("$\overline{f_{\mathrm{d}}}$")
axes[1].set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
axes[0].set_yticks([0.6, 1, 1.4])
axes[1].set_yticks([-4, 0, 4])
axes[1].set_xlabel("Local Time / h")
labels = ['CCIC Land', 'CCIC Ocean', 'ICON Control', 'ICON +2K', 'ICON +4K']
handles = [
    plt.Line2D([0], [0], color=color['land'], linestyle="-"),
    plt.Line2D([0], [0], color=color['sea'], linestyle="-"),
    plt.Line2D([0], [0], color="#462d7b", linestyle="--"),
    plt.Line2D([0], [0], color="#1f948a", linestyle="--"),
    plt.Line2D([0], [0], color="#c1df24", linestyle="--"),
]
fig.legend(handles, labels, frameon=False, ncols=1, bbox_to_anchor=(1.2, 0.6))
for ax in axes:
    ax.set_xlim([0, 24])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])

# add letters
for i, ax in enumerate(axes.flatten()):
    ax.text(
        0.02,
        1,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

fig.savefig("plots/thesis/diurnal_cycle.pdf", bbox_inches="tight", dpi=400)

# %% plots for talk --------------------------------------------
hist_icon_control = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon_hcap_data/control/production/daily_cycle_hist_2d.nc"
    )
    .coarsen(iwp=4, boundary="trim")
    .sum()
)
cf_icon = (
    hist_icon_control["hist"].sel(iwp=slice(1, None)).sum("iwp")
    / hist_icon_control["size"]
)
cf_icon = (cf_icon / cf_icon.sum("local_time")) * cf_ccic["sea"].mean("time").sum(
    "local_time"
)  # scale to gpm sea cf
#  plot diurnal cycle of both
fig, ax = plt.subplots(figsize=(6, 4))

for name in ["land", "sea"]:
    ax.plot(
        cf_ccic[name].local_time,
        cf_ccic[name].mean("time"),
        color=color[name],
        linestyle="-",
    )
    ax.plot(
        cf_gpm[name].local_time,
        cf_gpm[name].mean("time"),
        color=color[name],
        linestyle="--",
    )
# plot ICON
ax.plot(
    cf_icon.local_time,
    cf_icon.mean("time"),
    color="red",
)
ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("$f_{\mathrm{d}}$")
handles = [
    plt.Line2D([0], [0], color="green", linestyle="-"),
    plt.Line2D([0], [0], color="blue", linestyle="-"),
    plt.Line2D([0], [0], color="red", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="-"),
    plt.Line2D([0], [0], color="grey", linestyle="--"),
]
labels = ["Land", "Sea", "ICON", "$I$", r"$T_{\mathrm{b}}$"]

ax.legend(handles, labels, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([6, 12, 18])
ax.set_yticks([0.001, 0.002, 0.003])
ax.set_ylim(0.0006, 0.0032)
fig.savefig("plots/diurnal_cycle/talk/dc_icon.png", bbox_inches="tight", dpi=300)

# %% plot change of diurnal cycle
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
labels = {
    "all": "All",
    "sea": "Sea",
    "land": "Land",
}
for ax in axes:
    ax.axhline(0, color="black", linewidth=0.5)

for name in ["land", "sea"]:
    axes[1].plot(
        slopes_ccic[name].local_time,
        slopes_ccic[name],
        color=color[name],
        label=f"$I$ {labels[name]}",
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
        label=rf"$T_{{\mathrm{{b}}}} ~ \mathrm{{{labels[name]}}}$",
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
    slopes_icon["jed0022"],
    color="red",
    label="ICON +4K",
)
axes[0].plot(
    slopes_ccic["all"].local_time,
    slopes_ccic["all"],
    color="black",
    label=f"$I$ All",
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
    label=r"$T_{\mathrm{b}}$ All",
    linestyle="--",
)
axes[0].fill_between(
    slopes_gpm["all"].local_time,
    slopes_gpm["all"] - err_gpm["all"],
    slopes_gpm["all"] + err_gpm["all"],
    color="k",
    alpha=0.3,
)

for ax in axes:
    ax.set_xlim([0, 24])
    ax.set_xlabel("Local Time / h")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_yticks([-4, 0, 4])


axes[0].set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
fig.tight_layout()

fig.savefig("plots/diurnal_cycle/talk/change_all.png", dpi=300)

# %% plot for ap paper
temp_ccic = temps_deseason["sea"].sel(time=cf_ccic["sea"].time)
q90 = temp_ccic.quantile(0.5, dim="time")
q10 = temp_ccic.quantile(0.5, dim="time")
temp_90 = temp_ccic.where(temp_ccic >= q90, drop=True)
temp_10 = temp_ccic.where(temp_ccic <= q10, drop=True)
cf_90 = cf_ccic_deseason["sea"].sel(time=temp_90.time)
cf_10 = cf_ccic_deseason["sea"].sel(time=temp_10.time)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    cf_90.local_time,
    (cf_90.mean("time") - cf_10.mean("time"))
    / cf_ccic["sea"].mean("time")
    * 100
    / (temp_90.mean("time") - temp_10.mean("time")),
    color="k",
)
ax.plot(
    cf_90.local_time,
    slopes_ccic["sea"],
    color="blue",
)
ax.plot(
    cf_90.local_time,
    slopes_icon["jed0022"],
    color="red")
ax.set_xlim([0, 24])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("$f_{\mathrm{d}}$")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("plots/diurnal_cycle/talk/percentiles.png", dpi=300)
# %% calculate mean shift
change_ccic = slopes_ccic['sea']/100 * cf_ccic['sea'].mean('time')
change_icon = slopes_icon['jed0022']/100 * hists_icon['jed0022']
change_icon = xr.DataArray(list(change_icon), coords={'local_time':cf_ccic['sea'].local_time.values}, dims='local_time')
daytime_reduction_ccic = change_ccic.sel(local_time=slice(6,18)).sum() * 100 / cf_ccic['sea'].sel(local_time=slice(6,18)).mean('time').sum()
daytime_reduction_icon = change_icon.sel(local_time=slice(6,18)).sum() * 100 / cf_ccic['sea'].sel(local_time=slice(6,18)).mean('time').sum()
print(f"reduction daytime ccic: {daytime_reduction_ccic.values}")
print(f"reduction daytime icon: {daytime_reduction_icon.values}")


# %%
