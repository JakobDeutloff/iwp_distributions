import matplotlib.pyplot as plt

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