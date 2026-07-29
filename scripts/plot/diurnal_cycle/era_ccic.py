# %%
import xarray as xr
from src.helper_functions import (
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions
from scipy.signal import detrend
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.colors as mcolors

# %% load ccic and gpm data
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange", "icon": "green", "era5": "blue"}
names = ["ccic", "gpm", "era5"]
dim = {"ccic": "iwp", "gpm": "bt", "icon": "iwp", "era5": "iwp"}

hists = {}
hists["ccic"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
)
hists["gpm"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all.nc"
)
hists["era5"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_all.nc"
)

# %% calculate cloud fraction
cf = {}
for name in names:
    cf[name] = hists[name]["hist"] / hists[name]["size"]
# %% normalise cloud fraction
cf_norm = {}
for name in names:
    cf_norm[name] = cf[name] / cf[name].sum("local_time")

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %% regression long-term trend

def regress_hist_temp_2d_trend(cf, temp):
    if "bt" in cf.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    slopes = xr.zeros_like(cf.isel(time=0))
    p_values = xr.zeros_like(cf.isel(time=0))
    slope_temp, _, _, _, _ = linregress(
        np.arange(len(temp.sel(time=cf.time).values)), temp.sel(time=cf.time).values
    )
    for i in cf.local_time:
        for j in cf[detrend_dim]:
            cf_vals = cf.sel({"local_time": i, detrend_dim: j})
            cf_vals = cf_vals.where(np.isfinite(cf_vals), drop=True)
            slope_freq, _, _, p_value, _ = linregress(
                np.arange(len(cf_vals.values)), cf_vals.values
            )
            slopes.loc[{"local_time": i, detrend_dim: j}] = slope_freq / slope_temp
            p_values.loc[{"local_time": i, detrend_dim: j}] = p_value

    slopes_perc = slopes * 100 / cf.mean("time")  # convert to % / K
    return slopes_perc, p_values


slopes_lt = {}
p_values_lt = {}

for name in names:
    slopes_lt[name], p_values_lt[name] = regress_hist_temp_2d_trend(
        cf_norm[name].fillna(0), temp
    )

# %%  detrend and deseasonalize
cf_detrend = {}
temp_detrend = xr.DataArray(detrend(temp), coords=temp.coords, dims=temp.dims)
temp_detrend = deseason(temp_detrend)
for name in names:
    cf_detrend[name] = detrend_hist_2d(cf_norm[name])
    cf_detrend[name] = deseason(cf_detrend[name])

# %% regression internal variability
slopes_iv = {}
p_values_iv = {}

for name in names:
    slopes_iv[name], p_values_iv[name] = regress_hist_temp_2d(
        cf_detrend[name], temp_detrend, cf_norm[name]
    )

# %% calculate scaling of IWP bins
dist_ccic = hists["ccic"]["hist"].sum(["time", "local_time"]) / hists["ccic"][
    "size"
].sum("time")
dist_era5 = hists["era5"]["hist"].sum(["time", "local_time"]) / hists["era5"][
    "size"
].sum("time")
scaling = xr.DataArray(
    dist_era5.values / dist_ccic.values, coords=dist_era5.coords, dims=dist_era5.dims
)

# %% plot
fig, axes = plt.subplots(2, 2, figsize=(7, 9), sharey="row", sharex=True)


colors = ["#0E23E3", "white", "#FF0000"]
n_bins = 256
cmap_change = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", colors, N=n_bins
)

slopes = {
    "ccic_iv": slopes_iv["ccic"],
    "ccic_lt": slopes_lt["ccic"],
    "era5_iv": slopes_iv["era5"],
    "era5_lt": slopes_lt["era5"],
}
p_values = {
    "ccic_iv": p_values_iv["ccic"],
    "ccic_lt": p_values_lt["ccic"],
    "era5_iv": p_values_iv["era5"],
    "era5_lt": p_values_lt["era5"],
}

titles = {
    "ccic_iv": "CCIC Internal Variability",
    "ccic_lt": "CCIC Long-term Trend",
    "era5_iv": "ERA5 Internal Variability",
    "era5_lt": "ERA5 Long-term Trend",
}
scalings = {
    "ccic_iv": xr.DataArray(
        np.ones_like(slopes_iv["ccic"].iwp.values),
        coords=slopes_iv["ccic"].iwp.coords,
        dims=slopes_iv["ccic"].iwp.dims,
    ),
    "ccic_lt": xr.DataArray(
        np.ones_like(slopes_lt["ccic"].iwp.values),
        coords=slopes_lt["ccic"].iwp.coords,
        dims=slopes_lt["ccic"].iwp.dims,
    ),
    "era5_iv": scaling,
    "era5_lt": scaling,
}

for ax, (key, slope) in zip(axes.flatten(), slopes.items()):
    y_warped = (
        np.log10(slope.iwp.sel(iwp=slice(8e-2, 10)).values)
        * scalings[key].sel(iwp=slice(8e-2, 10)).values
    )  
    
    mask = p_values[key].sel(iwp=slice(8e-2, 10)).values > 0.05
    local_time_grid, dim_grid = np.meshgrid(
        p_values[key].local_time.values, y_warped, indexing="ij"
    )
    im = ax.pcolormesh(
        slope.local_time,
        y_warped,
        slope.sel(iwp=slice(8e-2, 10)).T,
        cmap=cmap_change,
        vmin=-6,
        vmax=6,
        rasterized=True,
    )

    ax.scatter(
        local_time_grid[mask],
        dim_grid[mask],
        color="black",
        marker="o",
        s=0.5,
        label="p > 0.05",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_ticks = [1, 0, -1]
    y_tick_idx = (
        np.searchsorted(np.log10(slope.iwp.sel(iwp=slice(8e-2, 10)).values), y_ticks)
        - 1
    )
    y_tick_locs = y_warped[y_tick_idx]
    ax.set_yticks(y_tick_locs)
    ax.set_yticklabels(["$10^{1}$", "$10^{0}$", "$10^{-1}$"])
    ax.set_xticks([6, 12, 18])
    ax.set_ylim(y_tick_locs[-1], y_tick_locs[0])

axes[0, 0].invert_yaxis()
axes[1, 0].invert_yaxis()
axes[0, 0].set_ylabel("$I$ / kg m$^{-2}$")
axes[1, 0].set_ylabel("$I$ / kg m$^{-2}$")
axes[1, 0].set_xlabel("Local Time / h")
axes[1, 1].set_xlabel("Local Time / h")

fig.colorbar(
    im,
    ax=axes,
    orientation="horizontal",
    label=r"$\dfrac{\mathrm{d}f}{f~\mathrm{d}T}$ / % K$^{-1}$",
    extend="both",
    aspect=40,
    pad=0.1,
)

# add letters
for ax, letter in zip(axes.flatten(), ["a", "b", "c", "d"]):
    ax.text(
        0.08,
        0.88,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
    )

# add text. Write CCIC and ERA5 on the left, and internal variability and long-term trend on the right
axes[0, 0].text(
    -0.4,
    0.5,
    "CCIC",
    fontsize=12,
    transform=axes[0, 0].transAxes,
    rotation=90,
    va="center",
)
axes[1, 0].text(
    -0.4,
    0.5,
    "ERA5",
    fontsize=12,
    transform=axes[1, 0].transAxes,
    rotation=90,
    va="center",
)
axes[0, 1].text(
    0.5,
    1.1,
    "Long-Term Trend",
    fontsize=12,
    transform=axes[0, 1].transAxes,
    ha="center",
)
axes[0, 0].text(
    0.5,
    1.1,
    "Internal Variability",
    fontsize=12,
    transform=axes[0, 0].transAxes,
    ha="center",
)

fig.savefig("plots/diurnal_cycle/long_term/trend_2d_ccic_era5.pdf", bbox_inches="tight")


# %% plot hists ccic and era5
fig, axes = plt.subplots(figsize=(6, 4))

axes.plot(
    hists["ccic"]["iwp"],
    (
        hists["ccic"]["hist"].sum(["time", "local_time"])
        / hists["ccic"]["size"].sum("time")
    ).values,
    label="CCIC",
    color="black",
)
axes.plot(
    hists["era5"]["iwp"],
    (
        hists["era5"]["hist"].sum(["time", "local_time"])
        / hists["era5"]["size"].sum("time")
    ).values,
    label="ERA5",
    color="blue",
)
axes.set_xscale("log")
axes.set_xlim([1e-2, 20])
axes.set_ylim([0, 0.015])
axes.spines[["top", "right"]].set_visible(False)
axes.set_xlabel(r"$I$ / kg m$^{-2}$")
axes.set_ylabel(r"$f(I)$")
axes.legend(frameon=False)

fig.savefig("plots/diurnal_cycle/long_term/hist_ccic_era5.pdf", bbox_inches="tight")

# %%
