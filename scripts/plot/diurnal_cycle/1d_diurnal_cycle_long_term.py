# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from src.helper_functions import (
    nan_detrend,
    deseason,
    regress_hist_temp_1d,
)
from scipy.signal import detrend
from scipy.stats import linregress


# %% load ccic data
color = {"all": "black", "sea": "blue", "land": "green"}
names = ["all", "sea", "land"]
dims = {"ccic": "iwp", "gpm": "bt"}
hists_ccic = {}
hists_gpm = {}
for name in names:
    hists_ccic[name] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_{name}.nc"
    )
    hists_gpm[name] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_{name}.nc"
    )


# %% calculate cloud fraction
cf_ccic = {}
cf_gpm = {}
for name in names:
    cf_ccic[name] = hists_ccic[name]["hist"].sel(iwp=slice(1, None)).sum(
        "iwp"
    ) / hists_ccic[name]["hist"].sum(["iwp", "local_time"])
    cf_gpm[name] = hists_gpm[name]["hist"].sel(bt=slice(None, 231)).sum(
        "bt"
    ) / hists_gpm[name]["hist"].sum(["bt", "local_time"])

# %% normalise cloud fraction
cf_ccic_norm = {}
cf_gpm_norm = {}
for name in names:
    cf_ccic_norm[name] = cf_ccic[name] / cf_ccic[name].sum("local_time")
    cf_gpm_norm[name] = cf_gpm[name] / cf_gpm[name].sum("local_time")

# %% load era5 surface temp
temps = {}
temps["all"] = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m
temps["sea"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc"
).t2m
temps["land"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_land.nc"
).t2m

# %% regression
slopes_ccic = {}
slopes_gpm = {}
err_ccic = {}
err_gpm = {}
for name in names:
    slopes_ccic[name], err_ccic[name] = regress_hist_temp_1d(
        cf_ccic[name], temps[name], cf_ccic[name]
    )
    slopes_gpm[name], err_gpm[name] = regress_hist_temp_1d(
        cf_gpm[name], temps[name], cf_gpm[name]
    )

# %%
def regress_hist_temp_1d_trend(cf, temp):

    slopes = xr.zeros_like(cf.isel(time=0))
    p_values = xr.zeros_like(cf.isel(time=0))
    slope_temp, _, _, _, _ = linregress(
        np.arange(len(temp.sel(time=cf.time).values)), temp.sel(time=cf.time).values
    )
    for i in cf.local_time:
            cf_vals = cf.sel({"local_time": i})
            cf_vals = cf_vals.where(np.isfinite(cf_vals), drop=True)
            slope_freq, _, _, p_value, _ = linregress(
                np.arange(len(cf_vals.values)), cf_vals.values
            )
            slopes.loc[{"local_time": i}] = slope_freq/slope_temp
            p_values.loc[{"local_time": i}] = p_value
    
    slopes_perc = slopes * 100 / cf.mean("time")  # convert to % / K
    return slopes_perc, p_values
slopes_linear_ccic = {}
slopes_linear_gpm = {}
p_values_linear_ccic = {}
p_values_linear_gpm = {}
for name in names:
     slopes_linear_ccic[name], p_values_linear_ccic[name] = regress_hist_temp_1d_trend(
        cf_ccic_norm[name].fillna(0), temps[name]
    )
     slopes_linear_gpm[name], p_values_linear_gpm[name] = regress_hist_temp_1d_trend(
        cf_gpm_norm[name].fillna(0), temps[name]
    )



# %% plot change of diurnal cycle from regression 
def plot_change_diurnal_cycle(slopes, err):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axhline(0, color="black", linewidth=0.5)
    for name in names:
        ax.plot(slopes[name].local_time, slopes[name], color=color[name], label=name)
        ax.fill_between(
            slopes[name].local_time,
            slopes[name] - err[name],
            slopes[name] + err[name],
            color=color[name],
            alpha=0.3,
        )

    ax.set_xlim([0, 24])
    ax.set_xlabel("Local Time / h")
    ax.set_ylabel(
        r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    fig.tight_layout()

    return fig


fig_ccic_change = plot_change_diurnal_cycle(slopes_ccic, err_ccic)
fig_gpm_change = plot_change_diurnal_cycle(slopes_gpm, err_gpm)

# %% plot change of diurnal cycle from linear trend
offset={
    'all': -17,
    'sea': -18,
    'land': -19,
}
line_labels = {
    'all': 'All',
    'sea': 'Ocean',
    'land': 'Land'
}

def plot_linear_trend(slope, p_value, ax):
    ax.axhline(0, color="black", linewidth=0.5)
    for name in names:
        ax.plot(slope[name].local_time, slope[name], color=color[name], label=line_labels[name])
        valid = np.where(p_value[name] < 0.05, offset[name], np.nan)
        ax.plot(p_value[name].local_time, valid, color=color[name], linewidth=4)

    ax.set_xlim([0, 24])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([6, 12, 18])
    ax.set_ylim([-20, 20])
    ax.set_xlabel("Local Time / h")
    ax.set_yticks([-10, 0, 10])

fig, axes = plt.subplots(1, 2,figsize=(10, 4), sharey=True)
fig_linear_ccic = plot_linear_trend(slopes_linear_ccic, p_values_linear_ccic, ax=axes[0])
fig_linear_gpm = plot_linear_trend(slopes_linear_gpm, p_values_linear_gpm, ax=axes[1])
axes[0].set_ylabel(
    r"$\dfrac{\mathrm{d}f_{\mathrm{d}}}{f_{\mathrm{d}}~\mathrm{d}T}$ / % K$^{-1}$"
)
axes[0].set_title("$I$")
axes[1].set_title("$T_{\mathrm{b}}$")

handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([0], [0], color="grey", linewidth=4))
labels.append("p < 0.05")
fig.legend(handles, labels, bbox_to_anchor=(0.7, -0.05), ncol=4, frameon=False)


# %% make plot of cf over sea 

fig, ax = plt.subplots(figsize=(5, 5))
ax.pcolormesh(
    cf_gpm["sea"].local_time,
    cf_gpm["sea"].time,
    cf_gpm["sea"] - cf_gpm["sea"].mean("time"),
    shading="auto",
    cmap="viridis",
)

# %%
