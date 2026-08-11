# %%
import matplotlib.pyplot as plt
import numpy as np
from src.plot import plot_regression, definitions
from src.helper_functions import load_histograms, load_slopes, load_cre


# %% load data
colors, line_labels, linestyles = definitions()
hists = load_histograms('all')
slopes, errors, pvals = load_slopes()
cre = load_cre()
models_control = ['icon_ap_control', 'rcemip_control', 'xshield_control', 'icon_amip_control']
models = ['icon_ap', 'rcemip', 'xshield', 'icon_amip']
obs = ['ccic', 'spare_ice',  'two_c_ice', 'dardar']

# %% plot all distributions and cre for 2016 
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False, height_ratios=[0.15, 3, 1])


# plot cloud type lables on ax0
position = {
    "Cirrus": 3e-3,
    "Anvil": 1e-1,
    "Deep Convection": 5}

axes[0].axvline(1e-2, color='k', linewidth=2)
axes[0].axvline(1e0, color='k', linewidth=2)
for label, xpos in position.items():
    axes[0].text(
        xpos,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=axes[0].get_xaxis_transform()
    )

for name in models_control:
    axes[1].plot(
        hists[name].iwp,
        hists[name],
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

for name in obs:
    axes[1].plot(
        hists[name].iwp,
        hists[name].sel(time='2016').mean('time'),
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

axes[1].set_ylim(0, 0.013)

axes[2].axhline(0, color="k", linewidth=0.5)
axes[2].plot(
    cre.iwp,
    cre['net'],
    color='k',
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-3, 2e1])
    ax.set_xscale("log")

axes[1].legend(frameon=False)
axes[1].set_ylabel(r"$P(I)$")
axes[2].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[2].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[2].set_yticks([-100, 0, 40])
axes[1].set_yticks([0, 0.006, 0.012])
axes[0].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].xaxis.set_major_locator(plt.NullLocator())
axes[0].xaxis.set_minor_locator(plt.NullLocator())

fig.tight_layout()
#add letters
for i, ax in enumerate(axes[1:]):
    ax.text(0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.savefig("plots/anvil_thinning/publication/distributions_cre_2016.pdf", bbox_inches="tight")
