# %%
import matplotlib.pyplot as plt
import numpy as np
from src.plot import definitions
from src.helper_functions import load_feedbacks, load_slopes

# %%
colors, line_labels, linestyles = definitions()
feedbacks, feedbacks_area, feedbacks_opacity = load_feedbacks()
slopes, errors, p_vals = load_slopes()
models = ['icon_ap', 'rcemip', 'xshield', 'icon_amip']
obs = ['ccic', 'spare_ice',  'two_c_ice', 'dardar']

# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])
markers = {
    "icon_ap": "x",
    "rcemip": "x",
    "xshield": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
    'icon_amip': "x",
}

for name in models + obs:
    axes[0].plot(
        feedbacks[name].iwp,
        feedbacks[name],
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

    axes[1].scatter(
        0,
        feedbacks[name].sum().item()/2,
        color=colors[name],
        marker=markers[name],
        label=line_labels[name],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        1,
        feedbacks_area[name].item()/2,
        color=colors[name],
        marker=markers[name],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        2,
        feedbacks_opacity[name].item()/2,
        color=colors[name],
        marker=markers[name],
        s=80,
        alpha=0.7
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False, loc="upper left", ncol=2)
axes[0].set_yticks([-0.02, 0, 0.02])
axes[0].set_ylim(-0.022, 0.03)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xlim(-0.5, 2.5)
axes[1].set_xticklabels(["Total", "Amount", "Optical \n Depth"], rotation=45)
axes[1].set_ylabel(r"$\lambda$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.02, 0, 0.05, 0.1])
axes[1].set_ylim(-0.04, 0.1)


handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(1.15, 0.98))

# add letters
for i, ax in enumerate(axes):
    ax.text(0.03, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig("plots/anvil_thinning/publication/feedback_monthly.pdf", bbox_inches="tight")

# %% calculate mean and std of feedback 
mean_feedback = np.mean([feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
std_feedback = np.std([feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
print(f"Mean feedback: {mean_feedback:.4f} W m^-2 K^-1")
print(f"Std feedback: {std_feedback:.4f} W m^-2 K^-1")
print(f"Feedback for ICON: {feedbacks['icon_ap'].sum().item()/2:.4f} W m^-2 K^-1")
print(f"Feedback for RCEMIP: {feedbacks['rcemip'].sum().item()/2:.4f} W m^-2 K^-1")
    
# %% caclculate feedback fro every satellite
total_feedback = [feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']]

# %% plot for thesis 
offsets = {
    "icon_ap_plus2K": 0.2,
    "icon_ap_plus4K": 0.3,
    "rcemip": 0.4,
    "ccic": 0.5,
    "two_c_ice": 0.6,
    "dardar": 0.7,
    "spare_ice": 0.8,
}
markers = {
    "icon_ap_plus2K": "x",
    "icon_ap_plus4K": "x",
    "rcemip": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
}
colors['icon_ap_plus2K'] = '#1f948a'
colors['icon_ap_plus4K'] = '#c1df24'
line_labels['icon_ap_plus2K'] = "ICON +2K"
line_labels['icon_ap_plus4K'] = "ICON +4K" 
linestyles['icon_ap_plus2K'] = "--"
linestyles['icon_ap_plus4K'] = "--"

fig, axes = plt.subplots(3, 2, figsize=(10, 8), height_ratios=[1, 0.3, 1], width_ratios=[1, 0.1], sharex='col')
axes[2, 0].axhline(0, color="k", linewidth=0.7)
axes[1, 0].axhline(0.05, color="k", linewidth=0.7)
# plot regression 
axes[0, 0].plot(
    slopes['icon_ap_plus2K'].iwp,
    slopes['icon_ap_plus2K'],
    label=line_labels['icon_ap_plus2K'],
    color=colors['icon_ap_plus2K'],
    linestyle="--",
)
axes[0, 0].plot(
    slopes['icon_ap_plus4K'].iwp,
    slopes['icon_ap_plus4K'],
    label=line_labels['icon_ap_plus4K'],
    color=colors['icon_ap_plus4K'],
    linestyle="--",
)


axes[0, 0].plot(
    slopes['rcemip'].iwp,
    slopes['rcemip'],
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)

for key in obs:
    axes[0, 0].plot(
        slopes[key].iwp,
        slopes[key],
        label=line_labels[key],
        color=colors[key],
    )
    axes[1, 0].plot(
        p_vals[key].iwp,
        p_vals[key],
        label=line_labels[key],
        color=colors[key],
    )

axes[0, 0].axhline(0, color="k", linewidth=0.5)

# plot feedback
members = offsets.keys()
for key in members:
    axes[2, 0].plot(
        feedbacks[key].iwp,
        feedbacks[key]/2,
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

    axes[2, 1].scatter(
        0,
        feedbacks[key].sum().item()/2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
    )

axes[0, 0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[0, 0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[1, 0].set_ylabel("p-value")
axes[1, 0].set_yticks([0.05, 0.5, 1])


axes[2, 0].set_ylabel(r"$\lambda_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[2, 0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[2, 0].set_yticks([-0.01, 0, 0.01])


axes[2, 1].spines[['bottom']].set_visible(False)
axes[2, 1].set_xticks([])
axes[2, 1].set_ylabel(r"$\lambda_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[2, 1].set_yticks([0, 0.05, 0.1, 0.15])
axes[2, 1].set_ylim([-0.035, 0.17])


for ax in axes[:, 0]: 
    ax.set_xlim(1e-3, 2e1)
    ax.set_xscale("log")
for ax in axes.flatten():
    ax.spines[["top", "right"]].set_visible(False)

# add letters
axes[0, 1].remove()
axes[1, 1].remove()
for i, ax in enumerate([axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]]):
    ax.text(0.02, 0.95, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout() 

# add legends 
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(0.99, 0.9))
handles, labels = axes[2, 1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(0.99, 0.7))
#fig.savefig("plots/thesis/feedback_monthly_thesis.pdf", bbox_inches="tight")
# %% print feedback values for table
for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar', 'icon_ap_plus2K', 'icon_ap_plus4K', 'rcemip']:
    print(f"{key}: {feedbacks[key].sum().item()/2:.3f} W m^-2 K^-1")