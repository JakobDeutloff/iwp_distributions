import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


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
        "dardar": "brown",
        "2c": "k",
        "ccic": "purple",
        "spare": "darkgreen",
    }

    labels = {
        "jed0011": "ICON Control",
        "jed0022": "ICON +4 K",
        "jed0033": "ICON +2 K",
        "rcemip": "RCEMIP",
        "dardar": "DARDAR",
        "2c": "2C-ICE",
        "ccic": "CCIC",
        "spare": "SPARE-ICE",
    }

    linestyles = {
        "jed0011": "--",
        "jed0022": "--",
        "jed0033": "--",
        "rcemip": "--",
        "dardar": "-",
        "2c": "-",
        "ccic": "-",
        "spare": "-",
    }

    return colors, labels, linestyles


def plot_2d_trend(area_fraction, slopes, area_change, feedback, p_values, feedback_cum, err_cum, dim):

    fig, axes = plt.subplots(2, 5, figsize=(12, 4), sharey='row', height_ratios=[1, 0.05], width_ratios=[2, 2, 2, 2, 1])

    # create colormaps
    # Create a diverging colormap: blue -> white -> red
    colors = ["#0E23E3", "white", "#FF0000"]
    n_bins = 256
    cmap_change = mcolors.LinearSegmentedColormap.from_list(
        "custom_diverging", colors, N=n_bins
    )

    colors = ["#0EC7E3", "white", "#FB06BA"]
    cmap_feedback = mcolors.LinearSegmentedColormap.from_list(
        "custom_diverging_feedback", colors, N=n_bins
    )

    # Get mask of where p_value > 0.05
    mask = p_values.values > 0.05
    local_time_grid, dim_grid = np.meshgrid(
        p_values.local_time.values, p_values[dim].values, indexing="ij"
    )

    # plot slopes
    im_slope = axes[0, 0].pcolor(
        slopes.local_time,
        slopes[dim],
        slopes.T,
        cmap=cmap_change,
        vmin=-6,
        vmax=6,
        rasterized=True,
    )
    axes[0, 0].scatter(
        local_time_grid[mask],
        dim_grid[mask],
        color="black",
        marker="o",
        s=0.5,
        label="p > 0.05",
    )

    # plot area fraction
    im_hist = axes[0, 1].pcolor(
        area_fraction.local_time,
        area_fraction[dim],
        area_fraction.T,
        cmap="binary_r",
        vmin=0,
        vmax=0.0005,
        rasterized=True,
    )

    # plot area change
    im_weighted = axes[0, 2].pcolor(
        area_change.local_time,
        area_change[dim],
        area_change.T,
        cmap=cmap_change,
        vmin=-7e-6,
        vmax=7e-6,
        rasterized=True,
    )
    axes[0, 2].scatter(
        local_time_grid[mask],
        dim_grid[mask],
        color="black",
        marker="o",
        s=0.5,
        label="p > 0.05",
    )

    # plot feedback
    im_feedback = axes[0, 3].pcolor(
        feedback.local_time,
        feedback[dim],
        feedback.T,
        cmap=cmap_feedback,
        vmin=-0.006,
        vmax=0.006,
        rasterized=True,
    )
    axes[0, 3].scatter(
        local_time_grid[mask & (local_time_grid >=6) & (local_time_grid <=18)],
        dim_grid[mask & (local_time_grid >=6) & (local_time_grid <=18)],
        color="black",
        marker="o",
        s=0.5,
        label="p > 0.05",
    )

    # plot cumsum of feedback
    axes[0, 4].plot(
        feedback_cum,
        feedback_cum[dim],
        color="k",
        label="Cumulative Feedback",
    )
    axes[0, 4].fill_betweenx(
        feedback_cum[dim],
        feedback_cum - err_cum,
        feedback_cum + err_cum,
        color="gray",
        alpha=0.5,
    )
    axes[0, 4].spines[["top", "right"]].set_visible(False)
    axes[0, 4].set_xlabel(r"$\sum_{I}$ $\lambda$ / W m$^{-2}$ K$^{-1}$")
    axes[0, 4].set_xticks([0, np.round(feedback_cum.isel({dim:-1}).values, 2)])

    if dim == "bt":
        for ax in axes[0,:-1]:
            ax.set_xlabel("Local Time / h")
            ax.set_ylim([200, 260])
            ax.set_yticks([200, 230, 260])
        axes[0, 0].set_ylabel(r"$T_{\mathrm{b}}$ / K")
    else:
        for ax in axes[0,:-1]:
            ax.set_yscale("log")
            ax.invert_yaxis()
            ax.set_ylim([10, 1e-1])
        axes[0, 0].set_ylabel("$I$ / kg m$^{-2}$")

    for ax in axes[0, :-1]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([6, 12, 18])
        ax.set_xlabel("Local Time / h")

    cb1 = fig.colorbar(
        im_slope,
        cax=axes[1, 0],
        label=r"$\dfrac{\mathrm{d}f}{f~\mathrm{d}T}$ / % K$^{-1}$",
        extend="both",
        orientation="horizontal",
    )
    cb1.set_ticks([-6, 0, 6])
    cb2 = fig.colorbar(
        im_hist,
        cax=axes[1, 1],
        label="$f$",
        extend="max",
        orientation="horizontal",
    )
    cb2.set_ticks([0, 0.00025, 0.0005])
    cb3 = fig.colorbar(
        im_weighted,
        cax=axes[1, 2],
        label=r"$\dfrac{\mathrm{d}f}{\mathrm{d}T}$ / K$^{-1}$",
        extend="both",
        orientation="horizontal",
    )
    cb3.set_ticks([-7e-6, 0, 7e-6])
    cb4 = fig.colorbar(
        im_feedback,
        cax=axes[1, 3],
        label="$\lambda$ / W m$^{-2}$ K$^{-1}$",
        extend="both",
        orientation="horizontal",
    )
    cb4.set_ticks([-0.006, 0, 0.006])
    axes[1, 4].remove()

    # add letters 
    for ax, letter in zip(axes[0, :], ["a", "b", "c", "d", "e"]):
        ax.text(
            0.08,
            0.88,
            letter,
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig, axes
