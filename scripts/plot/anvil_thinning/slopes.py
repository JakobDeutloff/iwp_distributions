# %%
import matplotlib.pyplot as plt
import numpy as np
from src.plot import plot_regression, definitions
from src.helper_functions import load_histograms, load_slopes, load_cre

# %% load data
colors, line_labels, linestyles = definitions()
hists = load_histograms("all")
slopes, errors, pvals = load_slopes()
cre = load_cre()
models = ["icon_ap", "rcemip", "xshield", "icon_amip"]
obs = ["ccic", "spare_ice", "two_c_ice", "dardar"]
# %% plot slopes and p-value
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])

for name in models + obs:
    axes[0].plot(
        slopes[name].iwp,
        slopes[name],
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

for name in obs:
    axes[1].plot(
        pvals[name].iwp,
        pvals[name],
        label=line_labels[name],
        color=colors[name],
    )

axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xscale("log")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1e-3, 2e1)

axes[0].set_ylabel(r"$\dfrac{\mathrm{d}P(I)}{\mathrm{d}T}$ / K$^{-1}$")
axes[1].set_ylabel("p-value")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[0].set_ylim(-0.00063, 0.0002)
axes[1].set_yticks([0.05, 0.5, 1])
axes[1].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, bbox_to_anchor=(0.85, 0))

# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight="bold"
    )

fig.savefig("plots/anvil_thinning/publication/slopes_monthly.pdf", bbox_inches="tight")


# %% plot relative change in slopes

fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(0, color="k", linewidth=0.5)

rel_slopes = {}
for name in models:
    rel_slopes[name] = (slopes[name] / hists[name + "_control"]) * 100
for name in obs:
    rel_slopes[name] = (slopes[name] / hists[name].mean("time")) * 100


for name in models + obs:
    ax.plot(
        slopes[name].iwp,
        rel_slopes[name],
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )


ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(1e-3, 2e1)
ax.set_ylim([-15, 10])
ax.set_ylabel(r"$\dfrac{\mathrm{d}P(I)}{P(I)\mathrm{d}T}$ / % K$^{-1}$")
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, bbox_to_anchor=(0.85, -0.05))

fig.savefig("plots/anvil_thinning/publication/slopes_relative.pdf", bbox_inches="tight")

# %% give numbers and ratios of relative slopes
rel_slopes_min = {}
for name in models + obs:
    rel_slopes_min[name] = rel_slopes[name].sel(iwp=slice(1e-3, 3)).min().values
    print(f"{name}: {rel_slopes_min[name]:.2f} % K^-1")

# %%
print(
    f"Max ratio: {np.min([rel_slopes_min[name] for name in obs]) / np.max([rel_slopes_min[name] for name in models]):.2f}"
)
print(
    f"Min ratio: {np.max([rel_slopes_min[name] for name in obs]) / np.min([rel_slopes_min[name] for name in models]):.2f}"
)

# %%
