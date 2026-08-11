# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import load_slopes, load_cre
from scipy.optimize import curve_fit
import pickle

# %% initialize containers
colors, line_labels, _ = definitions()
slopes_monthly, _, _ = load_slopes()
cre = load_cre()
models = ['icon_ap', 'rcemip', 'xshield', 'icon_amip']
obs = ['ccic', 'spare_ice',  'two_c_ice', 'dardar']
linestyles = [':', '-', '--']

# %% fit sigmoid function
def sigmoid_cre(log_iwp, k1, logI01, k2, logI02, A):
    return (
        A
        * (0.5 - (1 + np.exp(-k1 * (log_iwp - logI01))) ** -1)
        * (1 + np.exp(-k2 * (log_iwp - logI02))) ** -1
    )


x = np.log10(cre.sel(iwp=slice(1e-3, 20))["iwp"].values)
y = cre.sel(iwp=slice(1e-3, 20))["net"].values

# remove non-finite values
mask = np.isfinite(x) & np.isfinite(y)
x_fit = x[mask]
y_fit = y[mask]

# initial guesses
k1 = 3
logI0_1 = x_fit[np.argmin(np.abs(y_fit))]
k2 = 2
logI0_2 = -1.4
A = 200
p0 = [k1, logI0_1, k2, logI0_2, A]

res = curve_fit(sigmoid_cre, x_fit, y_fit, p0=p0, ftol=1e-6, maxfev=10000)
print("Fitted parameters:", res[0])

# %% shift within range of crossover values
cre_shifted = {}
range_crossover = [-0.3, 0, 0.3]
for i in range(3):
    crossover = range_crossover[i]
    cre_shift = sigmoid_cre(np.log10(cre["iwp"]) - crossover, *res[0])
    cre_shift = xr.DataArray(
        cre_shift,
        coords={"iwp": cre["iwp"]},
        dims=["iwp"],
    )
    cre_shifted[i] = cre_shift

# %% calculate feedback
feedback_up = {}
feedback_mid = {}
feedback_low = {}
for key in slopes_monthly.keys():
    feedback_up[key] = slopes_monthly[key] * cre_shifted[0].values
    feedback_mid[key] = slopes_monthly[key] * cre_shifted[1].values
    feedback_low[key] = slopes_monthly[key] * cre_shifted[2].values

# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])

for key in models + obs:
    axes[0].plot(
        feedback_up[key].iwp,
        feedback_up[key],
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[0],
    )
    axes[0].plot(
        feedback_mid[key].iwp,
        feedback_mid[key],
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[1],
    )
    axes[0].plot(
        feedback_low[key].iwp,
        feedback_low[key],
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[2],
    )

    axes[1].scatter(
        0,
        feedback_up[key].sum().item() / 2,
        color=colors[key],
        marker="x",
        label=line_labels[key],
    )
    axes[1].scatter(
        1,
        feedback_mid[key].sum().item() / 2,
        color=colors[key],
        marker="o",
        label=line_labels[key],
    )
    axes[1].scatter(
        2,
        feedback_low[key].sum().item() / 2,
        color=colors[key],
        marker="^",
        label=line_labels[key],
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].set_yticks([-0.02, 0, 0.02])

axes[1].set_xticks([0, 1, 2])
axes[1].set_xlim(-0.5, 2.5)
axes[1].set_xticklabels(["$I_0 = 0.1$", "$I_0 = 0.2$", "$I_0 = 0.4$"], rotation=45)
axes[1].set_ylabel(r"$\lambda_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.02, 0, 0.05, 0.1])


# %% make one combined plot of C(I) and integrated feedback 
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])

# plot C(I)
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].plot(cre["iwp"], cre["net"], color="grey", label="Original", linewidth=4)
for i in range(3):
    crossover = 10 ** (res[0][1] + range_crossover[i])
    axes[0].plot(
        cre_shifted[i].iwp,
        cre_shifted[i].values,
        label=f"Fit, $I_0={crossover:.1f}$",
        linestyle=linestyles[i],
        color="k",
    )
axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 20)
axes[0].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False)
axes[0].set_yticks([-100, 0, 40])

# plot feedback
axes[1].axhline(0, color="k", linewidth=0.5)
markers = {
    "icon_ap": "x",
    "rcemip": "x",
    "xshield": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
    "icon_amip": "x"
}
for key in models + obs:
    axes[1].scatter(
        0,
        feedback_up[key].sum().item() / 2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        1,
        feedback_mid[key].sum().item() / 2,
        color=colors[key],
        marker=markers[key],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        2,
        feedback_low[key].sum().item() / 2,
        color=colors[key],
        marker=markers[key],
        s=80,
        alpha=0.7
    )

axes[1].set_xticks([0, 1, 2])
axes[1].set_xlim(-0.5, 2.5)
axes[1].set_xticklabels(["$I_0 = 0.1$", "$I_0 = 0.2$", "$I_0 = 0.4$"], rotation=45)
axes[1].set_ylabel(r"$\lambda$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.05, 0, 0.05, 0.1, 0.15])

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(1.15, 0.98))

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

# add letters
for i, ax in enumerate(axes.flatten()):
    ax.text(0.03, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig('plots/anvil_thinning/publication/cre_sensitivity.pdf', bbox_inches='tight')
# %%
