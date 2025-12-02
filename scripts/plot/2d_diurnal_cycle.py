# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
from dask.diagnostics import ProgressBar
from src.helper_functions import nan_detrend
from src.plot import definitions
from scipy.signal import detrend
from scipy.stats import linregress

# %%
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gridsat": "red", "gpm": "orange"}
bins = {
    "ccic": np.arange(0, 25, 1),
    "gridsat": np.arange(0, 27, 3),
    "gpm": np.arange(0, 25, 1),
}
names = ["ccic", "gridsat", "gpm"]


# %% open gridsat
def process_gridsat(ds):
    ds = ds.sortby("time")
    return ds


hist_gridsat = xr.open_mfdataset(
    "/work/bm1183/m301049/gridsat/coarse/gridsat_2d_hist*.nc",
    preprocess=process_gridsat,
).load()
hist_gridsat = hist_gridsat.coarsen(bt=2, boundary="trim").sum()

# %% open gpm
hist_gpm = xr.open_mfdataset("/work/bm1183/m301049/GPM_MERGIR/hists/gpm_*.nc").load()
hist_gpm = hist_gpm.coarsen(bt=2, boundary="trim").sum()

# %% open ccic
hist_ccic = xr.open_mfdataset(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d_*.nc"
).load()
hist_ccic = hist_ccic.coarsen(iwp=4, boundary="trim").sum()
# %%
hists = {
    "gridsat": hist_gridsat,
    "gpm": hist_gpm,
    "ccic": hist_ccic,
}


# %%
def resample_histograms(hist):
    hist_m = hist.resample(time="1ME").sum()
    hist_m["time"] = pd.to_datetime(hist_m["time"].dt.strftime("%Y-%m"))
    hist_m = hist_m["hist"] / hist_m["hist"].sum("local_time")
    return hist_m


hists_monthly = {}
for name in names:
    hists_monthly[name] = resample_histograms(hists[name])

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files = [f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 1980]
ds = xr.open_mfdataset(files, engine="cfgrib", combine="by_coords")


def get_montly_temp(mask):
    with ProgressBar():
        temp = (
            ds["t2m"]
            .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30) & mask)
            .mean("values")
            .compute()
        )
    temp["time"] = pd.to_datetime(temp["time"].dt.strftime("%Y-%m"))
    return temp


temp = get_montly_temp(True)


# %%  detrend and deseasonalize
def detrend_temp(t):
    t_detrend = xr.DataArray(detrend(t), coords=t.coords, dims=t.dims)
    t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
        "time"
    )
    t_deseason["time"] = pd.to_datetime(t_deseason["time"].dt.strftime("%Y-%m"))
    return t_deseason


def detrend_hist(hist):

    out = xr.zeros_like(hist)

    if "bt" in hist.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    for i in hist[detrend_dim]:
        hist_detrend = nan_detrend(hist.sel({detrend_dim: i}), dim="local_time")
        out.loc[{detrend_dim: i}] = hist_detrend
    hist_deseason = out.groupby("time.month") - out.groupby("time.month").mean("time")
    hist_deseason["time"] = pd.to_datetime(hist_deseason["time"].dt.strftime("%Y-%m"))
    return hist_deseason


hists_detrend = {}
for name in names:
    temp_detrend = detrend_temp(temp)
    hists_detrend[name] = detrend_hist(hists_monthly[name])

# %% regression
slopes = {}
p_values = {}


def regress_hist_temp(hist, temp):
    if "bt" in hist.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    slopes = xr.zeros_like(hist.isel(time=0))
    p_values = xr.zeros_like(hist.isel(time=0))
    for i in hist.local_time:
        for j in hist[detrend_dim]:
            hist_vals = hist.sel({"local_time": i, detrend_dim: j})
            hist_vals = hist_vals.where(np.isfinite(hist_vals), drop=True)
            temp_vals = temp.sel(time=hist_vals.time)
            slope, intercept, r_value, p_value, std_err = linregress(
                temp_vals.values, hist_vals.values
            )
            slopes.loc[{"local_time": i, detrend_dim: j}] = slope
            p_values.loc[{"local_time": i, detrend_dim: j}] = p_value
    return slopes, p_values


for name in names:
    slopes[name], p_values[name] = regress_hist_temp(hists_detrend[name], temp_detrend)
    slopes[name] = (
        slopes[name] * 100 / hists_monthly[name].mean("time")
    )  # convert to %/K

# %%


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


# %% plot slopes ccic
mean_hists = {}
mean_hists["ccic"] = hists["ccic"]["hist"].sel(iwp=slice(1e-3, 10)).sum("time") / hists[
    "ccic"
]["size"].sum("time")
fig, axes = plot_2d_trend(
    mean_hists["ccic"] / mean_hists["ccic"].max(),
    slopes["ccic"].sel(iwp=slice(1e-3, 10)),
    p_values["ccic"].sel(iwp=slice(1e-3, 10)),
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/ccic_2d_trend_3.png", dpi=300)

# %% plot slopes gpm
mean_hists["gpm"] = hists["gpm"]["hist"].sel(bt=slice(190, 260)).sum("time") / hists[
    "gpm"
]["size"].sum("time")
fig, axes = plot_2d_trend(
    mean_hists["gpm"] / mean_hists["gpm"].max(),
    slopes["gpm"].sel(bt=slice(190, 260)),
    p_values["gpm"].sel(bt=slice(190, 260)),
    dim="bt",
)
fig.savefig("plots/diurnal_cycle/gpm_2d_trend_3.png", dpi=300)


# %% plot slopes gridsat
mean_hists["gridsat"] = hists["gridsat"]["hist"].sel(bt=slice(190, 260)).sum(
    "time"
) / hists["gridsat"]["size"].sum("time")
fig, axes = plot_2d_trend(
    mean_hists["gridsat"] / mean_hists["gridsat"].max(),
    slopes["gridsat"].sel(bt=slice(190, 260)),
    p_values["gridsat"].sel(bt=slice(190, 260)),
    dim="bt",
)
fig.savefig("plots/diurnal_cycle/gridsat_2d_trend.png", dpi=300)


# %% plot mean diurnal cycle
mean_ccic = (
    hists["ccic"].sel(iwp=slice(1e-1, 10)).sum(["time", "iwp"])["hist"]
    / hists["ccic"].sel(iwp=slice(1e-1, 10))["hist"].sum()
)
mean_gpm = (
    hists["gpm"].sel(bt=slice(190, 230)).sum(["time", "bt"])["hist"]
    / hists["gpm"].sel(bt=slice(190, 230))["hist"].sum()
)
mean_gridsat = (
    hists["gridsat"].sel(bt=slice(190, 230)).sum(["time", "bt"])["hist"]
    / hists["gridsat"].sel(bt=slice(190, 230))["hist"].sum()
    / 3
)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mean_ccic["local_time"], mean_ccic, label="CCIC", color=color["ccic"])
ax.plot(
    mean_gridsat["local_time"], mean_gridsat, label="Gridsat", color=color["gridsat"]
)
ax.plot(mean_gpm["local_time"], mean_gpm, label="GPM", color=color["gpm"])
ax.set_ylabel("Normalized Histogram")

# %% calculate feedback
SW_in = xr.open_dataarray(
    "/work/bm1183/m301049/icon_hcap_data/publication/incoming_sw/SW_in_daily_cycle.nc"
)
SW_in = SW_in.interp(time_points=slopes["ccic"]["local_time"], method="linear")
rad_changes = {}
area_changes = {}
feedbacks = {}
# %% ccic feedback
area_changes["ccic"] = (slopes["ccic"].sel(iwp=slice(1e-1, 10)) / 100) * mean_hists[
    "ccic"
]  # 1 / K
rad_changes["ccic"] = (area_changes["ccic"] * SW_in * 0.7) - (
    (area_changes["ccic"]) * SW_in * 0.3
)  # W / m^2 / K
feedbacks["ccic"] = rad_changes["ccic"].sum()
print(f"CCIC cloud feedback: {feedbacks['ccic'].values:.2f} W/m2/K")

# %% gpm feedback
area_changes["gpm"] = (slopes["gpm"].sel(bt=slice(190, 260)) / 100) * mean_hists[
    "gpm"
]  # 1 / K
rad_changes["gpm"] = (area_changes["gpm"] * SW_in * 0.7) - (
    (area_changes["gpm"]) * SW_in * 0.3
)  # W / m^2 / K
feedbacks["gpm"] = rad_changes["gpm"].sum()
print(f"GPM cloud feedback: {feedbacks['gpm'].values:.2f} W/m2/K")

# %% gridsat feedback
area_changes["gridsat"] = (
    slopes["gridsat"].sel(bt=slice(190, 230)) / 100
) * mean_hists[
    "gridsat"
]  # 1 / K
rad_changes["gridsat"] = (area_changes["gridsat"] * SW_in * 0.7) - (
    (area_changes["gridsat"]) * SW_in * 0.3
)  # W / m^2 / K
feedbacks["gridsat"] = rad_changes["gridsat"].sum()
print(f"Gridsat cloud feedback: {feedbacks['gridsat'].values:.2f} W/m2/K")
# %% plot cumulative area change
fig, ax = plt.subplots(figsize=(8, 5))

# %%
