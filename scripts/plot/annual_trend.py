# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, definitions
from src.helper_functions import nan_detrend
import pickle


# %% initialize containers
hists_monthly = {}
hists_smooth = {}
slopes_monthly = {}
error_montly = {}
bins = bins = np.logspace(-3, 2, 254)[::4]
datasets = ["ccic", "2c", "dardar", "spare"]
colors, line_labels, linestyles = definitions()

# %% open CCIC
path = "/work/bm1183/m301049/ccic/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hists_ccic = xr.concat(hist_list, dim="time")

# %% open 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% open dardar
hists_dardar = xr.open_dataset("/work/bm1183/m301049/dardarv3.10/hist_dardar.nc")

# %% open spareice
hists_spare = xr.open_dataset("/work/bm1183/m301049/spareice/hists_metop.nc")

# %% coarsen histograms and normalise by size
hists_ccic_coarse = hists_ccic.coarsen(bin_center=4, boundary="trim").sum()

# %% calculate annual means
hists_annual = {}

#  ccic
hists_ccic_annual = hists_ccic_coarse.resample(time="1YE").sum()
hists_annual["ccic"] = hists_ccic_annual["hist"] / hists_ccic_annual["size"]

# 2c-ice
good_years = ["2007", "2008", "2009", "2010", "2013", "2014", "2015", "2016", "2017"]
hists_2c_annual = hists_2c.resample(time="1YE").sum()
mask = np.isin(hists_2c_annual["time"].dt.year.astype(str), good_years)
mask_good_years = xr.DataArray(
    mask, dims=["time"], coords={"time": hists_2c_annual["time"]}
)
hists_2c_annual = hists_2c_annual.where(mask_good_years)
hists_annual["2c"] = hists_2c_annual["hist"] / hists_2c_annual["size"]

# dardar
hists_dardar_annual = hists_dardar.resample(time="1YE").sum().where(mask_good_years)
hists_annual["dardar"] = hists_dardar_annual["hist"] / hists_dardar_annual["size"]

# spareice
hists_spare_annual = hists_spare.resample(time="1YE").sum()
hists_annual["spare"] = hists_spare_annual["hist"] / hists_spare_annual["size"]


# %% load temperature data from ERA5
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

#  slect tropics and calculate annual average
with ProgressBar():
    t_month = (
        ds["t2m"]
        .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30))
        .mean("values")
        .compute()
    )

t_month["time"] = pd.to_datetime(t_month["time"].dt.strftime("%Y-%m"))
t_annual = t_month.resample(time="1YE").mean("time")

# %% detrend
hists_detrend = {}
for ds in datasets:
    hists_detrend[ds] = nan_detrend(hists_annual[ds])

t_detrend = xr.DataArray(detrend(t_annual), coords=t_annual.coords, dims=t_annual.dims)
# %% calculate regression
slopes = {}
error = {}

for ds in datasets:
    slopes_ds = []
    err_ds = []
    hist_vals = hists_detrend[ds].where(hists_detrend[ds].notnull(), drop=True)
    temp = t_detrend.sel(time=hist_vals.time)
    for i in range(hists_detrend[ds].bin_center.size):
        hist_row = hist_vals.isel(bin_center=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
    slopes[ds] = xr.DataArray(
        slopes_ds,
        coords={"bin_center": hists_detrend[ds].bin_center},
        dims=["bin_center"],
    )
    error[ds] = xr.DataArray(
        err_ds,
        coords={"bin_center": hists_detrend[ds].bin_center},
        dims=["bin_center"],
    )
# %% save slopes
with open("/work/bm1183/m301049/iwp_dists/slopes_annual.pkl", "wb") as f:
    pickle.dump(slopes, f)
with open("/work/bm1183/m301049/iwp_dists/error_annual.pkl", "wb") as f:
    pickle.dump(error, f)

# %% plot annual variability and regression slopes
fig_ccic, axes_ccic = plot_regression(
    t_detrend.sel(time=hists_annual["ccic"].time),
    hists_detrend["ccic"].T,
    slopes["ccic"],
    error["ccic"],
    "CCIC Annual",
)
fig_ccic.savefig("plots/ccic_annual.png", dpi=300, bbox_inches="tight")

# %%
fig_2c, axes_2c = plot_regression(
    t_detrend.sel(time=hists_annual["2c"].time),
    hists_detrend["2c"].T,
    slopes["2c"],
    error["2c"],
    "2C-ICE Annual",
)
fig_2c.savefig("plots/2c_annual.png", dpi=300, bbox_inches="tight")
# %%
fig_dardar, axes_dardar = plot_regression(
    t_detrend.sel(time=hists_annual["dardar"].time),
    hists_detrend["dardar"].T,
    slopes["dardar"],
    error["dardar"],
    "DARDAR Annual",
)
fig_dardar.savefig("plots/dardar_annual.png", dpi=300, bbox_inches="tight")

# %%
fig_spare, axes_spare = plot_regression(
    t_detrend.sel(time=hists_annual["spare"].time),
    hists_detrend["spare"].T,
    slopes["spare"],
    error["spare"],
    "SPARE-ICE Annual",
)
fig_spare.savefig("plots/spare_annual.png", dpi=300, bbox_inches="tight")
# %% plot all slopes together
fig, ax = plt.subplots(figsize=(8, 5))
for ds in datasets:
    mean_hist = hists_annual[ds].mean("time")
    ax.plot(
        slopes[ds]["bin_center"],
        slopes[ds],
        label=line_labels[ds],
        color=colors[ds],
        linestyle=linestyles[ds],
        linewidth=2,
    )
ax.set_xscale("log")
ax.set_xlim([1e-3, 40])
ax.axhline(0, color="k", linestyle="--", linewidth=1)
# %% load cre 
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)

# interpolate
cre["iwp"] = np.log10(cre["iwp"])
cre = cre.interp(
    iwp=np.log10(hists_annual["ccic"].bin_center), method="linear"
).drop_vars("iwp")
cre['bin_center'] = 10 ** cre['bin_center']
# %% calculate feedback
feedback = {}
for ds in datasets:
    feedback[ds] = slopes[ds] * cre["net"].values

# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[3, 1])
offsets = {
    "ccic": 0,
    "2c": 0.1,
    "dardar": 0.2,
    "spare": 0.3,
}

for ds in datasets:
    axes[0].plot(
        hists_annual[ds].bin_center,
        feedback[ds],
        label=line_labels[ds],
        color=colors[ds],
        linestyle=linestyles[ds],
    )

    axes[1].scatter(
        offsets[ds],
        feedback[ds].sum().item(),
        color=colors[ds],
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel("$F_{\mathrm{IWP}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel("I / kg m$^{-2}$")
axes[0].legend()
axes[1].set_xticks(list(offsets.values()))
labels = [line_labels[ds] for ds in list(offsets.keys())]
axes[1].set_xticklabels(labels, rotation=45, ha="right")
axes[1].set_ylabel("$F_{\mathrm{IWP}}$ / W m$^{-2}$ K$^{-1}$")
fig.tight_layout()
fig.savefig("plots/feedback_annual.png", dpi=300, bbox_inches="tight")


# %%
