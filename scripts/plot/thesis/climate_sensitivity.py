# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
colors = {
    "+2K": "#1f948a",
    "+4K": "#c1df24",
    "rcemip": "#ff7f0e",
    "DARDAR": "brown",
    "2C-ICE": "k",
    "CCIC": "purple",
    "SPARE-ICE": "darkgreen",
    "sherwood": "#0077b6",
    "deutloff": "#f348e8",
}

# %% define estimates and uncertainty
sherwood_estimates = {"altitude": 0.1, "amount": -0.2}
sherwood_uncertainty = {"altitude": 0.05, "amount": 0.2}

icon_estimates = {
    "altitude": {"+2K": 0.164, "+4K": 0.126},
    "amount": {"+2K": -0.024, "+4K": 0.002},
    "diurnal_cycle": {"+2K": 0.104, "+4K": 0.052},
}
rcemip_estimates = {"amount": 0.026}
satellite_estimates = {
    "amount": {"CCIC": 0.038, "SPARE-ICE": 0.067, "2C-ICE": 0.045, "DARDAR": 0.056},
    "diurnal_cycle": {"CCIC": 0.12},
}
satellite_uncertainty = {"diurnal_cycle": 0.02}

# %%
icon_mean_estimates = {
    "altitude": np.mean(list(icon_estimates["altitude"].values())),
    "amount": np.mean(list(icon_estimates["amount"].values())),
    "diurnal_cycle": np.mean(list(icon_estimates["diurnal_cycle"].values())),
}
satellite_mean_estimates = {
    "amount": np.mean(list(satellite_estimates["amount"].values())),
    "diurnal_cycle": np.mean(list(satellite_estimates["diurnal_cycle"].values())),
}

# %% calculate total feedback sherwood
sherwood_estimates["total"] = (
    sherwood_estimates["altitude"] + sherwood_estimates["amount"]
)
sherwood_uncertainty["total"] = (
    sherwood_uncertainty["altitude"] ** 2 + sherwood_uncertainty["amount"] ** 2
) ** 0.5

deutloff_estimate = {
    "altitude": icon_mean_estimates["altitude"],
    "amount": np.mean(
        [
            icon_mean_estimates["amount"],
            rcemip_estimates["amount"],
            satellite_mean_estimates["amount"],
        ]
    ),
    "diurnal_cycle": np.mean(
        [
            icon_mean_estimates["diurnal_cycle"],
            satellite_mean_estimates["diurnal_cycle"],
        ]
    ),
}

deutloff_errors = {
    "altitude": np.abs(
        icon_estimates["altitude"]["+2K"] - icon_estimates["altitude"]["+4K"]
    )
    / 2,
    "amount": np.abs(
        np.min(
            [
                icon_estimates["amount"]["+2K"],
                icon_estimates["amount"]["+4K"],
                rcemip_estimates["amount"],
                satellite_estimates["amount"]["CCIC"],
                satellite_estimates["amount"]["2C-ICE"],
                satellite_estimates["amount"]["DARDAR"],
                satellite_estimates["amount"]["SPARE-ICE"],
            ]
        )
        - np.max(
            [
                icon_estimates["amount"]["+2K"],
                icon_estimates["amount"]["+4K"],
                rcemip_estimates["amount"],
                satellite_estimates["amount"]["CCIC"],
                satellite_estimates["amount"]["2C-ICE"],
                satellite_estimates["amount"]["DARDAR"],
                satellite_estimates["amount"]["SPARE-ICE"],
            ]
        )
    )
    / 2,
    "diurnal_cycle": np.abs(
        np.min(
            [
                icon_estimates["diurnal_cycle"]["+2K"],
                icon_estimates["diurnal_cycle"]["+4K"],
                satellite_estimates["diurnal_cycle"]["CCIC"]
                - satellite_uncertainty["diurnal_cycle"],
            ]
        )
        - np.max(
            [
                icon_estimates["diurnal_cycle"]["+2K"],
                icon_estimates["diurnal_cycle"]["+4K"],
                satellite_estimates["diurnal_cycle"]["CCIC"]
                + satellite_uncertainty["diurnal_cycle"],
            ]
        )
    )
    / 2,
}
deutloff_estimate["total"] = (
    deutloff_estimate["altitude"]
    + deutloff_estimate["amount"]
    + deutloff_estimate["diurnal_cycle"]
)
deutloff_errors["total"] = np.sqrt(
    deutloff_errors["altitude"] ** 2
    + deutloff_errors["amount"] ** 2
    + deutloff_errors["diurnal_cycle"] ** 2
)

# %% plot sherwood only
fig, ax = plt.subplots(figsize=(4, 3))

ax.errorbar(
    sherwood_estimates["altitude"],
    0,
    xerr=sherwood_uncertainty["altitude"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)
ax.errorbar(
    sherwood_estimates["amount"],
    -1,
    xerr=sherwood_uncertainty["amount"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)
ax.errorbar(
    sherwood_estimates["total"],
    -2,
    xerr=sherwood_uncertainty["total"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)


ax.axvline(0, color="k", linewidth=0.5)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_yticks([0, -1, -2])
ax.set_yticklabels(["High-Cloud Altitude", "Tropical Anvil Amount", "Total"])
ax.get_yticklabels()[-1].set_weight("bold")
ax.set_xlabel(r"$\lambda_{\mathrm{hc}}$ / W m$^{-2}$ K$^{-1}$")
fig.savefig("plots/thesis/feedback_estimates_sherwood.pdf", bbox_inches="tight")


# %% plot own estimates with sheerwood
fig, ax = plt.subplots(figsize=(9, 3))

# sherwood
ax.errorbar(
    sherwood_estimates["altitude"],
    0,
    xerr=sherwood_uncertainty["altitude"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)
ax.errorbar(
    sherwood_estimates["amount"],
    -1,
    xerr=sherwood_uncertainty["amount"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)
ax.errorbar(
    sherwood_estimates["total"],
    -3,
    xerr=sherwood_uncertainty["total"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["sherwood"],
)

# rcemip
ax.scatter(
    rcemip_estimates["amount"], -1.2, marker="x", color=colors["rcemip"], label="RCEMIP", alpha=0.7
)

# satellites
for dataset in ["CCIC", "2C-ICE", "DARDAR", "SPARE-ICE"]:
    ax.scatter(
        satellite_estimates["amount"][dataset],
        -1.2,
        marker="o",
        color=colors[dataset],
        label=dataset,
    )

ax.errorbar(
    satellite_mean_estimates["diurnal_cycle"],
    -2.2,
    xerr=satellite_uncertainty["diurnal_cycle"],
    marker="o",
    color=colors["CCIC"],
    capsize=5,
    capthick=2,
    zorder=1
)

# icon
for run in ["+2K", "+4K"]:
    ax.scatter(
        icon_estimates["altitude"][run],
        -0.2,
        marker="x",
        color=colors[run],
        label=f"ICON {run}",
    )
    ax.scatter(
        icon_estimates["amount"][run],
        -1.2,
        marker="x",
        color=colors[run],
        label=f"ICON {run}",
    )
    ax.scatter(
        icon_estimates["diurnal_cycle"][run],
        -2.2,
        marker="x",
        color=colors[run],
        label=f"ICON {run}",
        zorder=2,
    )

# my estimates
ax.errorbar(
    deutloff_estimate["altitude"],
    -0.4,
    xerr=deutloff_errors["altitude"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["deutloff"],
)
ax.errorbar(
    deutloff_estimate["amount"],
    -1.4,
    xerr=deutloff_errors["amount"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["deutloff"],
)
ax.errorbar(
    deutloff_estimate["diurnal_cycle"],
    -2.4,
    xerr=deutloff_errors["diurnal_cycle"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["deutloff"],
)
ax.errorbar(
    deutloff_estimate["total"],
    -3.2,
    xerr=deutloff_errors["total"],
    fmt="d",
    capsize=5,
    capthick=2,
    color=colors["deutloff"],
)

ax.axvline(0, color="k", linewidth=0.7)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_yticks([-0.2, -1.2, -2.2, -3.1])
ax.set_yticklabels(
    ["High-Cloud Altitude", "Tropical Anvil Amount", "Tropical Diurnal Cycle", "Total"]
)
ax.tick_params(axis="y", length=0)
ax.axhline(-0.7, color="grey", linewidth=0.5, linestyle="--")
ax.axhline(-1.7, color="grey", linewidth=0.5, linestyle="--")
ax.axhline(-2.7, color="grey", linewidth=0.5, linestyle="--")

# make total bold
ax.get_yticklabels()[-1].set_weight("bold")
ax.set_xlabel(r"$\lambda_{\mathrm{hc}}$ / W m$^{-2}$ K$^{-1}$")

# make legend
handles = [
    plt.Line2D([0], [0], marker="d", color=colors["sherwood"], label="Sherwood et al. (2020)", linestyle="-"),
    plt.Line2D([0], [0], marker="d", color=colors["deutloff"], label="Deutloff et al.", linestyle="-"),
    plt.Line2D([0], [0], marker="x", color=colors["+2K"], label="ICON + 2K", linestyle=""),
    plt.Line2D([0], [0], marker="x", color=colors["+4K"], label="ICON + 4K", linestyle=""),
    plt.Line2D([0], [0], marker="x", color=colors["rcemip"], label="RCEMIP", linestyle=""),
    plt.Line2D([0], [0], marker="o", color=colors["CCIC"], label="CCIC", linestyle=""),
    plt.Line2D([0], [0], marker="o", color=colors["2C-ICE"], label="2C-ICE", linestyle=""),
    plt.Line2D([0], [0], marker="o", color=colors["DARDAR"], label="DARDAR", linestyle=""),
    plt.Line2D([0], [0], marker="o", color=colors["SPARE-ICE"], label="SPARE-ICE", linestyle=""),
]
ax.legend(handles=handles, bbox_to_anchor=(0.8, -0.2), ncols=3, frameon=False)
fig.savefig("plots/thesis/feedback_estimates.pdf", bbox_inches="tight")

# %% calculate revised climate sensitivity
feedback = -1.3
feedback_revised = feedback + (deutloff_estimate["total"] - sherwood_estimates["total"])
forcing = 4
sensitivity = -forcing / feedback
sensitivity_revised = -forcing / feedback_revised

print(f"Original sensitivity: {sensitivity:.2f} K")
print(f"Revised sensitivity: {sensitivity_revised:.2f} K")


# %% print best estimate values for table 
for key in ["altitude", "amount", "diurnal_cycle", "total"]:
    print(f"{key.capitalize()}: {deutloff_estimate[key]:.2f} ± {deutloff_errors[key]:.2f} W m^-2 K^-1")

# %%
