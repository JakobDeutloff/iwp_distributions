# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import re
from dask.diagnostics import ProgressBar
from src.helper_functions import nan_detrend, shift_longitudes, read_ccic_dc
from src.plot import definitions
from scipy.signal import detrend
from scipy.stats import linregress

# %%
colors, line_labels, linestyles = definitions()

# %% open CCIC
hists = {}
names = ['all', 'sea', 'land', 'gridsat']
hists['sea'] = read_ccic_dc("ccic_cpcir_daily_cycle_distribution_sea_")
hists['all'] = read_ccic_dc("ccic_cpcir_daily_cycle_distribution_")
hists['land'] = hists['all'] - hists['sea']

# %% open gridsat
def process_gridsat(ds):
    ds = ds.sortby("time")
    return ds
hist_gridsat = xr.open_mfdataset("/work/bm1183/m301049/gridsat/hourly/gridsat_2d_hist_*.nc", preprocess=process_gridsat).load()

# %%
hists['gridsat'] = hist_gridsat.sel(bt=slice(None, 220)).sum('bt')
hists['gridsat']['size'] = hists['gridsat']['hist'].sum('local_time')

# %%
mask = (
    xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")
    .load()
    .pipe(shift_longitudes, lon_name="lon")
)

# %%
def resample_histograms(hist):
    hist_monthly = hist.resample(time="1ME").sum()
    hist_monthly["time"] = pd.to_datetime(hist_monthly["time"].dt.strftime("%Y-%m"))
    hist_monthly = hist_monthly["hist"] / hist_monthly["size"]
    hist_monthly = hist_monthly.transpose("local_time", "time")
    return hist_monthly

hists_monthly = {}
for name in names:
    hists_monthly[name] = resample_histograms(hists[name])

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 1980
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

mask_trop = mask.sel(
    lat=ds.isel(time=0).latitude, lon=ds.isel(time=0).longitude, method="nearest"
)

temps = {}
masks = {
    'all': True,
    'sea': mask_trop,
    'land': ~mask_trop,
    'gridsat': True,
}

def get_montly_temp(mask):
    with ProgressBar():
        temp = (
            ds["t2m"]
            .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30) & mask)
            .mean('values')
            .compute()
        )
    temp["time"] = pd.to_datetime(temp["time"].dt.strftime("%Y-%m"))
    return temp

for name in names:
    temps[name] = get_montly_temp(masks[name])


# %%  detrend and deseasonalize

def detrend_temp(t):
    t_detrend = xr.DataArray(detrend(t), coords=t.coords, dims=t.dims)
    t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby(
        "time.month"
    ).mean("time")
    t_deseason["time"] = pd.to_datetime(t_deseason["time"].dt.strftime("%Y-%m"))
    return t_deseason

def detrend_hist(hist):
    hist_detrend = nan_detrend(hist, dim="local_time")
    hist_deseason = hist_detrend.groupby("time.month") - hist_detrend.groupby(
        "time.month"
    ).mean("time")
    hist_deseason["time"] = pd.to_datetime(hist_deseason["time"].dt.strftime("%Y-%m"))
    return hist_deseason

for name in names:
    temps[name] = detrend_temp(temps[name])
    hists_monthly[name] = detrend_hist(hists_monthly[name])

# %% regression
slopes = {}
err = {}

def regress_hist_temp(hist, temp):
    slopes = []
    err = []
    hist_dummy = hist.where(hist.notnull(), drop=True)
    temp_vals = temp.sel(time=hist_dummy.time).values
    for i in range(hist_dummy.local_time.size):
        hist_vals = hist_dummy.isel(local_time=i).values
        slope, intercept, r_value, p_value, std_err = linregress(temp_vals, hist_vals)
        slopes.append(slope)
        err.append(std_err)
    slopes_da = xr.DataArray(
        slopes,
        coords={"local_time": hist_dummy.local_time},
        dims=["local_time"],
    )
    err_da = xr.DataArray(
        err,
        coords={"local_time": hist_dummy.local_time},
        dims=["local_time"],
    )
    return slopes_da, err_da

for name in names:
    slopes[name], err[name] = regress_hist_temp(hists_monthly[name], temps[name])

# %% load icon
runs = ["jed0011", "jed0022", "jed0033"]
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
hists_icon = {}
hists_raw = {}
for run in runs:
    hists_raw[run] = xr.open_dataset(
        f"/work/bm1183/m301049/icon_hcap_data/publication/distributions/{run}_deep_clouds_daily_cycle.nc"
    )
    hists_icon[run] = (hists_raw[run].sum("day") / hists_raw[run].sum())[
        "__xarray_dataarray_variable__"
    ].values


change_icon = {}
for run in runs[1:]:
    change_icon[run] = (hists_icon[run] - hists_icon["jed0011"]) / temp_delta[run]

# %% plot of mean daily cycle
mean_sea = hists['sea'].sum("time")


fig, ax = plt.subplots(figsize=(8, 5))
ax.stairs(
    mean_sea["hist"] / mean_sea["size"],
    np.arange(0, 25, 1),
    label="CCIC Sea",
    color='darkblue',
    linewidth=2,
)
for run in runs:
    ax.stairs(
        hists_icon[run], np.arange(0, 25, 1), label=line_labels[run], color=colors[run]
    )
ax.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax.set_ylim([0.03, 0.055])
ax.set_yticks([0.03, 0.04, 0.05])
ax.set_xlim([0, 23.9])
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_mean.png", dpi=300, bbox_inches="tight")

# %% plot mean daily cycle all regions 
fig, ax = plt.subplots(figsize=(8, 5))
color = {'all': 'black', 'sea': 'blue', 'land': 'green', 'gridsat': 'red'}
bins = {
    'all': np.arange(0, 25, 1),
    'sea': np.arange(0, 25, 1),
    'land': np.arange(0, 25, 1),
    'gridsat': np.arange(0, 25, 1),
}

for name in names[:-1]:
    mean_hist = hists[name].sum("time")
    ax.stairs(
        mean_hist["hist"] / mean_hist["size"],
        bins[name],
        label=f"CCIC {name}",
        color=color[name],
        linewidth=2,
    )

ax.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax.set_ylim([0.02, 0.075])
ax.set_xlim([0, 23.9])
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_mean_all_regions.png", dpi=300, bbox_inches="tight")


# %% plot change in diurnal cycle
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', linewidth=0.5)
ccic_region = 'all'
mean_ccic = hists[ccic_region].sum("time")["hist"] / hists[ccic_region].sum("time")["size"]

ax.plot(
    slopes[ccic_region]["local_time"],
    slopes[ccic_region] * 100 / mean_ccic,
    label="CCIC",
    color=colors["ccic"],
    linewidth=2,
)


ax.fill_between(
    slopes[ccic_region]["local_time"],
    slopes[ccic_region] * 100 / mean_ccic - err[ccic_region] * 100 / mean_ccic,
    slopes[ccic_region] * 100 / mean_ccic + err[ccic_region] * 100 / mean_ccic,
    alpha=0.3,
    color=colors["ccic"],
)

for run in runs[1:]:
    ax.plot(
        slopes[ccic_region]["local_time"],
        change_icon[run] * 100 / hists_icon["jed0011"],
        label=line_labels[run],
        color=colors[run],
    )
ax.set_ylabel("dP(I $>$ 1 kg m$^{-2}$)/dT / % K$^{-1}$")
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([0, 23.9])
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_change.png", dpi=300, bbox_inches="tight")

# %% plot change in diurnal cycle all regions
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', linewidth=0.5)

for name in names[:-1]:
    mean_ccic = hists[name].sum("time")["hist"] / hists[name].sum("time")["size"]
    ax.plot(
        slopes[name]["local_time"],
        slopes[name] * 100 / mean_ccic,
        label=f"CCIC {name}",
        color=color[name],
        linewidth=2,
    )

    ax.fill_between(
        slopes[name]["local_time"],
        slopes[name] * 100 / mean_ccic - err[name] * 100 / mean_ccic,
        slopes[name] * 100 / mean_ccic + err[name] * 100 / mean_ccic,
        alpha=0.3,
        color=color[name],
    )

ax.set_ylabel("dP(I $>$ 1 kg m$^{-2}$)/dT / % K$^{-1}$")
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([0, 23.9])
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_change_all_regions.png", dpi=300, bbox_inches="tight")

# %% plot change in diurnal cycle gridsat vs ccic
fig, ax = plt.subplots(figsize=(8, 5))
ax.axhline(0, color='k', linewidth=0.5)

for name in ['all', 'gridsat']:
    mean_ccic = hists[name].sum("time")["hist"] / hists[name].sum("time")["size"]
    ax.plot(
        slopes[name]["local_time"],
        slopes[name] * 100 / mean_ccic,
        label=f"CCIC {name}",
        color=color[name],
        linewidth=2,
    )

    ax.fill_between(
        slopes[name]["local_time"],
        slopes[name] * 100 / mean_ccic - err[name] * 100 / mean_ccic,
        slopes[name] * 100 / mean_ccic + err[name] * 100 / mean_ccic,
        alpha=0.3,
        color=color[name],
    )

ax.set_ylabel("dP(I $>$ 1 kg m$^{-2}$)/dT / % K$^{-1}$")
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([0, 23.9])
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_change_gridsat.png", dpi=300, bbox_inches="tight")


# %% plot mean daily cycle gridsat vs ccic
fig, ax = plt.subplots(figsize=(8, 5))
color = {'all': 'black', 'sea': 'blue', 'land': 'green', 'gridsat': 'red'}
bins = {
    'all': np.arange(0, 25, 1),
    'sea': np.arange(0, 25, 1),
    'land': np.arange(0, 25, 1),
    'gridsat': np.arange(0, 25, 1),
}

for name in ['all', 'gridsat']:
    mean_hist = hists[name].sum("time")
    ax.stairs(
        mean_hist["hist"] / mean_hist["size"],
        bins[name],
        label=f"CCIC {name}",
        color=color[name],
        linewidth=2,
    )

ax.set_ylabel("P($I$ > 1 kg m$^{-2}$)")
ax.set_ylim([0.02, 0.075])
ax.set_xlim([0, 23.9])
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_mean_gridsat.png", dpi=300, bbox_inches="tight")

# %%
