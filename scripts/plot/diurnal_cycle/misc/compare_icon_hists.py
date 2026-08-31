# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load icon
runs = ["jed0011", "jed0022", "jed0033"]
names = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
}
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
colors = {
    "jed0011": "black",
    "jed0022": "red",
    "jed0033": "blue",
}
hists_1d = {}
hists_2d = {}
for run in runs:
    hists_1d[run] = (
        xr.open_dataarray(
            f"/work/bu1562/m301049/icon_hcap_data/publication/distributions/{run}_deep_clouds_daily_cycle.nc"
        )
        .sum("day")
        .rename({"local_hour": "local_time"})
    )
    hists_2d[run] = xr.open_dataset(
        f"/work/bu1562/m301049/icon_hcap_data/{names[run]}/production/daily_cycle_hist_2d.nc"
    ).sum("time")
    hists_1d[run]["local_time"] = hists_2d[run]["local_time"]

# %% check if absolute number of samples is similar
for run in runs:
    total_samples_1d = hists_1d[run].sum().values
    total_samples_2d = hists_2d[run]["hist"].sel(iwp=slice(1e0, None)).sum().values
    print(
        f"Run {run} ({names[run]}): Total samples 1D = {total_samples_1d}, Total samples 2D = {total_samples_2d}"
    )

# %% calculate 1d hists from 2d hists
hists_1d_from_2d = {}
for run in runs:
    hist_2d = hists_2d[run]["hist"]
    hists_1d_from_2d[run] = hist_2d.sel(iwp=slice(1e0, None)).sum("iwp")

# %% plot normalised hists
fig, ax = plt.subplots(figsize=(8, 6))
for run in runs:
    norm_hist_1d = hists_1d[run] / hists_1d[run].sum()
    norm_hist_2d = hists_1d_from_2d[run] / hists_1d_from_2d[run].sum()
    ax.plot(
        norm_hist_1d["local_time"],
        norm_hist_1d,
        label=f"1D {names[run]}",
        linestyle="-",
        color=colors[run],
    )
    ax.plot(
        norm_hist_2d["local_time"],
        norm_hist_2d,
        label=f"2D {names[run]}",
        linestyle="--",
        color=colors[run],
    )

# %% calculate dc change from 2d hists
hists_2d_norm = {}
relative_change = {}
absolute_change = {}

for run in runs:
    hists_2d_norm[run] = hists_2d[run]["hist"] / hists_2d[run]["hist"].sum("local_time")

for run in runs[1:]:
    relative_change[run] = ((hists_2d_norm[run] - hists_2d_norm[runs[0]]) * 100) / (
        temp_delta[run] * hists_2d_norm[runs[0]]
    )  # % / K

for run in runs[1:]:
    absolute_change[run] = (relative_change[run] / 100) * (
        hists_2d["jed0011"]["hist"] / hists_2d["jed0011"]["hist"].sum()
    )


# %% plot 2d hists
fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharey=True, sharex=True)


for i, run in enumerate(runs[1:]):
    # relative change
    im_rel = axes[i, 0].pcolormesh(
        hists_2d[run]["local_time"],
        hists_2d[run]["iwp"],
        relative_change[run].T,
        rasterized=True,
        cmap="bwr",
        vmin=-3,
        vmax=3,
    )

    axes[i, 1].invert_yaxis()
    axes[i, 0].set_yscale("log")
    axes[i, 0].set_ylim([10, 1e-1])

    # absolute change
    im_abs = axes[i, 1].pcolormesh(
        hists_2d[run]["local_time"],
        hists_2d[run]["iwp"],
        absolute_change[run].T,
        rasterized=True,
        cmap="PRGn",
        vmax=1.5e-6,
        vmin=-1.5e-6,
    )

    axes[i, 1].invert_yaxis()
    axes[i, 1].set_yscale("log")
    axes[i, 1].set_ylim([10, 1e-1])

axes[1, 1].set_xlabel("Local Time / h")
axes[1, 0].set_xlabel("Local Time / h")
axes[0, 0].set_ylabel("$I$ / kg m$^{-2}$")
axes[1, 0].set_ylabel("$I$ / kg m$^{-2}$")
fig.colorbar(
    im_rel,
    ax=axes[:, 0],
    label=r"$\frac{\mathrm{d}P}{P\mathrm{d}T}$ / % K$^{-1}$",
    orientation="horizontal",
    pad=0.1,
)
fig.colorbar(
    im_abs,
    ax=axes[:, 1],
    label=r"$\frac{\mathrm{d}P}{\mathrm{d}T}$ / K$^{-1}$",
    orientation="horizontal",
    pad=0.1,
)
# %%
