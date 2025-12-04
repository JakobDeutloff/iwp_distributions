import matplotlib.pyplot as plt
import numpy as np


def plot_regression(temp, hists, slopes, error, title):
    fig, axes = plt.subplots(
        2, 3, figsize=(8, 8), sharex="col", width_ratios=[6, 24, 1]
    )

    # temp
    axes[0, 0].plot(temp.values, temp.time, color="black")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("T / K")
    axes[0, 0].set_ylabel("Year")
    axes[0, 0].set_ylim([hists.time.max(), hists.time.min()])
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    # iwp anomalies
    im = axes[0, 1].pcolormesh(
        hists.bin_center,
        hists.time,
        (hists - hists.mean("time")).T,
        cmap="seismic",
        vmin=-0.001,
        vmax=0.001,
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlim([1e-3, 2e1])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_ylim([hists.time.max(), hists.time.min()])
    axes[0, 1].set_yticklabels([])

    # slopes
    axes[1, 1].plot(hists.bin_center, slopes, color="k", label="CCIC")
    axes[1, 1].fill_between(
        hists.bin_center,
        slopes - error,
        slopes + error,
        color="gray",
        alpha=0.5,
        label="95% confidence interval",
    )
    axes[1, 1].axhline(0, color="gray", linestyle="-")
    axes[1, 1].set_ylabel("dP(I)/dT / K$^{-1}$")
    axes[1, 1].set_xlabel("I / kg m$^{-2}$")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    axes[1, 0].remove()
    axes[1, 2].remove()
    fig.colorbar(im, cax=axes[0, 2], label="P(I) anomaly")

    fig.suptitle(title, y=0.95)
    return fig, axes


def plot_hists(hists, temp, bins):
    temp = temp.sel(time=hists.time)
    norm = plt.Normalize(vmin=temp.min(), vmax=temp.max())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    for t in hists.time:
        axes[0].stairs(
            hists.sel(time=t),
            bins,
            alpha=0.7,
            color=cmap(norm(temp.sel(time=t).values)),
        )
        axes[1].stairs(
            hists.sel(time=t) - hists.mean("time"),
            bins,
            alpha=0.7,
            color=cmap(norm(temp.sel(time=t).values)),
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim([1e-3, 2e1])
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("P(I)")
    axes[1].set_ylabel("P(I) anomaly")
    axes[1].set_xlabel("I / kg m$^{-2}$")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label="Temperature / K")
    return fig


def definitions():

    colors = {
        "jed0011": "#462d7b",
        "jed0022": "#c1df24",
        "jed0033": "#1f948a",
        "rcemip": "#ff7f0e",
        'dardar': 'brown',
        '2c': 'k', 
        'ccic': 'purple',
        'spare': 'darkgreen'
    }

    labels = {
        'jed0011': 'ICON Control', 
        'jed0022': 'ICON +4 K',
        'jed0033': 'ICON +2 K',
        'rcemip': 'RCEMIP',
        'dardar': 'DARDAR',
        '2c': '2C-ICE',
        'ccic': 'CCIC',
        'spare': 'SPARE-ICE',
    }

    linestyles = {
        "jed0011": "--",
        "jed0022": "--",
        "jed0033": "--",
        "rcemip": "--",
        'dardar': '-',
        '2c': '-',
        'ccic': '-',
        'spare': '-',
    }


    return colors, labels, linestyles

def plot_2d_trend(mean_hist, slopes, p_values, dim):

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Get mask of where p_value > 0.05
    mask = p_values.values > 0.05
    local_time_grid, dim_grid = np.meshgrid(
        p_values.local_time.values, p_values[dim].values, indexing="ij"
    )

    # plot slopes
    im_slope = axes[0].pcolor(
        slopes.local_time,
        slopes[dim],
        slopes.T,
        cmap="seismic",
        vmin=-15,
        vmax=15,
    )
    axes[0].scatter(
        local_time_grid[mask],
        dim_grid[mask],
        color="black",
        marker="o",
        s=1,
        label="p > 0.05",
    )

    # plot mean histogram
    im_hist = axes[1].pcolor(
        mean_hist.local_time, mean_hist[dim], mean_hist.T, cmap="binary", vmin=0, vmax=1
    )

    # plot weighted sensitivity
    weighted = (slopes / 100) * mean_hist
    im_weighted = axes[2].pcolor(
        weighted.local_time,
        weighted[dim],
        weighted.T,
        cmap="seismic",
        vmin=-0.02,
        vmax=0.02,
    )
    axes[2].scatter(
        local_time_grid[mask],
        dim_grid[mask],
        color="black",
        marker="o",
        s=1,
        label="p > 0.05",
    )

    if dim == "bt":
        for ax in axes:
            axes[0].set_ylabel("Brightness Temperature / K")
            ax.set_xlabel("Local Time / h")

    else:
        for ax in axes:
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_xlabel("Local Time / h")
        axes[0].set_ylabel("$I$ / kg m$^{-2}$")

    fig.colorbar(
        im_slope,
        ax=axes[0],
        label="Sensitivity / % K$^{-1}$",
        extend="both",
        orientation="horizontal",
    )
    fig.colorbar(
        im_hist,
        ax=axes[1],
        label="Normalised Histogram",
        extend="neither",
        orientation="horizontal",
    )
    fig.colorbar(
        im_weighted,
        ax=axes[2],
        label="Weighted Sensitivity / K$^{-1}$",
        extend="both",
        orientation="horizontal",
    )
    fig.tight_layout()
    return fig, axes