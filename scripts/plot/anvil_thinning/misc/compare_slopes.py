# %%
import pickle
import matplotlib.pyplot as plt
from src.plot import definitions
import xarray as xr
import numpy as np

# %%
datasets = ["ccic", "2c", "dardar", "spare"]
colors, line_labels, linestyles = definitions()
# %% load slopes
with open("/work/bm1183/m301049/iwp_dists/slopes_monthly.pkl", "rb") as f:
    slopes_monthly = pickle.load(f)
with open("/work/bm1183/m301049/iwp_dists/error_monthly.pkl", "rb") as f:
    error_montly = pickle.load(f)
with open("/work/bm1183/m301049/iwp_dists/slopes_annual.pkl", "rb") as f:
    slopes_annual = pickle.load(f)
with open("/work/bm1183/m301049/iwp_dists/error_annual.pkl", "rb") as f:
    error_annual = pickle.load(f)
with open("/work/bm1183/m301049/iwp_dists/slopes_season.pkl", "rb") as f:
    slopes_season = pickle.load(f)
with open("/work/bm1183/m301049/iwp_dists/error_season.pkl", "rb") as f:
    error_season = pickle.load(f)

# %% load cre
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)
# interpolate
cre["iwp"] = np.log10(cre["iwp"])
cre = cre.interp(
    iwp=np.log10(slopes_monthly["ccic"].bin_center), method="linear"
).drop_vars("iwp")
cre["bin_center"] = 10 ** cre["bin_center"]
# %% calculate feedback
feedback_monthly = {}
feedback_annual = {}
feedback_season = {}
for ds in datasets:
    feedback_monthly[ds] = slopes_monthly[ds] * cre["net"].values
    feedback_annual[ds] = slopes_annual[ds] * cre["net"].values
    feedback_season[ds] = slopes_season[ds] * cre["net"].values
# %% plot
fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharey='col', sharex='col', width_ratios=[3, 3, 1])


for ds, ax1, ax2, ax3 in zip(datasets, axes[:, 0], axes[:, 1], axes[:, 2]):
    ax1.plot(
        slopes_monthly[ds]["bin_center"],
        slopes_monthly[ds],
        label=f"{line_labels[ds]} Monthly",
        color=colors[ds],
    )
    ax1.plot(
        slopes_annual[ds]["bin_center"],
        slopes_annual[ds],
        label=f"{line_labels[ds]} Annual",
        color=colors[ds],
        linestyle="--",
    )
    ax1.plot(
        slopes_season[ds]["bin_center"],
        slopes_season[ds],
        label=f"{line_labels[ds]} Seasonal",
        color=colors[ds],
        linestyle=":",
    )
    ax1.set_xscale("log")
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_xlim([1e-3, 40])
    ax1.set_ylabel('d$P(I)$/d$T$ / K$^{-1}$')
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(
        feedback_monthly[ds]["bin_center"],
        feedback_monthly[ds],
        label=f"{line_labels[ds]} Monthly",
        color=colors[ds],
    )

    ax2.plot(
        feedback_annual[ds]["bin_center"],
        feedback_annual[ds],
        label=f"{line_labels[ds]} Annual",
        color=colors[ds],
        linestyle="--",
    )
    ax2.plot(
        feedback_season[ds]["bin_center"],
        feedback_season[ds],
        label=f"{line_labels[ds]} Seasonal",
        color=colors[ds],
        linestyle=":",
    )
    ax2.set_xscale("log")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlim([1e-3, 40])
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylabel('$F_I(I)$ / W m$^{-2}$ K$^{-1}$')

    ax3.scatter(
        0,
        feedback_monthly[ds].sum().item(),
        label=f"{line_labels[ds]} Monthly",
        color=colors[ds],
        marker='o',
    )
    ax3.scatter(
        0,
        feedback_annual[ds].sum().item(),
        label=f"{line_labels[ds]} Annual",
        color=colors[ds],
        marker='x',
    )
    ax3.scatter(
        0,
        feedback_season[ds].sum().item(),
        label=f"{line_labels[ds]} Seasonal",
        color=colors[ds],
        marker='^',
    )
    ax3.set_ylim([0, 0.35])
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.set_xticks([])
    ax3.set_ylabel('$F_I$ / W m$^{-2}$ K$^{-1}$')

# make legend 
labels = [
    'CCIC', '2C-ICE', 'DARDAR', 'SPARE-ICE', 'Monthly', 'Annual', 'Seasonal', 'Monthly', 'Annual', 'Seasonal'
]
handles = [
    plt.Line2D([0], [0], color=colors['ccic']),
    plt.Line2D([0], [0], color=colors['2c']),
    plt.Line2D([0], [0], color=colors['dardar']),
    plt.Line2D([0], [0], color=colors['spare']),
    plt.Line2D([0], [0], color='k', linestyle='-'),
    plt.Line2D([0], [0], color='k', linestyle='--'),
    plt.Line2D([0], [0], color='k', linestyle=':'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8),
    plt.Line2D([0], [0], marker='x', color='k', markerfacecolor='k', markersize=8),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8),
]
fig.legend(handles, labels, ncol=3, bbox_to_anchor=(0.7, 0))
axes[-1, 0].set_xlabel('$I$ / kg m$^{-2}$')
axes[-1, 1].set_xlabel('$I$ / kg m$^{-2}$')
fig.tight_layout()
fig.savefig("plots/annual_monthly_feedback.png", dpi=300, bbox_inches="tight")

# %%
