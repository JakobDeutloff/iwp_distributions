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
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, interpolate_bins
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

# ccic
hists_monthly["ccic"] = (
    hists_ccic_coarse["hist"].resample(time="1ME").sum()
    / hists_ccic_coarse["size"].resample(time="1ME").sum()
)

# 2c-ice
hists_monthly["2c"] = (hists_2c["hist"] / hists_2c["size"]).sel(
    time=slice("2006-06", "2017-12")
)

# dardar
hists_monthly["dardar"] = (
    hists_dardar["hist"].resample(time="1ME").sum()
    / hists_dardar["size"].resample(time="1ME").sum()
).sel(time=slice("2006-06", "2017-12"))
hists_monthly["dardar"] = hists_monthly["dardar"].where(
    hists_dardar["size"].resample(time="1ME").sum() > 2.1e6
)

# spareice
hists_monthly["spare"] = hists_spare["hist"] / hists_spare["size"]
hists_monthly["spare"] = hists_monthly["spare"].transpose("bin_center", "time")
hists_monthly["spare"] = hists_monthly["spare"].sel(time=slice(None, "2025-07"))


# %% fix time coordinates
for key in datasets:
    hists_monthly[key]["time"] = pd.to_datetime(
        hists_monthly[key]["time"].dt.strftime("%Y-%m")
    )

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords").chunk({'time': 1, 'values':-1})

# slect tropics and calculate annual average
with ProgressBar():
    t_month = (
        ds["t2m"]
        .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30))
        .mean("values")
        .compute()
    )


t_month["time"] = pd.to_datetime(t_month["time"].dt.strftime("%Y-%m"))
t_annual = t_month.resample(time="1YE").mean("time")

# %%
plot_hists(
    hists_monthly["spare"].sel(time=slice("2007-05", "2025-07")),
    t_month.sel(time=slice("2007-05", "2025-07")),
    bins,
)

# %% detrend and deseasonalize monthly values
hists_deseason = {}

# temperature
t_detrend = xr.DataArray(detrend(t_month), coords=t_month.coords, dims=t_month.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)

for ds in datasets:
    hists_detrend = nan_detrend(hists_monthly[ds])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[ds] = hists_deseason_ds

# %%regression
slopes_ccic = []
err_ccic = []
for ds in datasets:
    slopes_ds = []
    err_ds = []
    hist_vals = hists_deseason[ds].where(hists_deseason[ds].notnull(), drop=True)
    temp = t_deseason.sel(time=hist_vals.time)
    for i in range(hists_deseason[ds].bin_center.size):
        hist_row = hist_vals.isel(bin_center=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
    slopes_monthly[ds] = xr.DataArray(
        slopes_ds,
        coords={"bin_center": hists_deseason[ds].bin_center},
        dims=["bin_center"],
    )
    error_montly[ds] = xr.DataArray(
        err_ds,
        coords={"bin_center": hists_deseason[ds].bin_center},
        dims=["bin_center"],
    )
# %% calculate seasonal means
hists_season = {}
t_season = t_detrend.groupby("time.month").mean("time")
for key in datasets:
    hists_season[key] = hists_monthly[key].groupby("time.month").mean(
        "time"
    ) - hists_monthly[key].mean("time")

# %% regression seasonal means
slopes_season = {}
error_season = {}
for key in datasets:
    slopes = []
    error = []
    for i in range(hists_season[key].bin_center.size):
        hist_vals = hists_season[key].isel(bin_center=i).values
        res = linregress(t_season.values, hist_vals)
        slopes.append(res.slope)
        error.append(res.stderr)
    slopes_season[key] = xr.DataArray(
        slopes,
        coords={"bin_center": hists_season[key].bin_center},
        dims=["bin_center"],
    )
    error_season[key] = xr.DataArray(
        error,
        coords={"bin_center": hists_season[key].bin_center},
        dims=["bin_center"],
    )
# %% save slopes
with open("/work/bm1183/m301049/iwp_dists/slopes_monthly.pkl", "wb") as f:
    pickle.dump(slopes_monthly, f)
with open("/work/bm1183/m301049/iwp_dists/error_monthly.pkl", "wb") as f:
    pickle.dump(error_montly, f)
with open("/work/bm1183/m301049/iwp_dists/slopes_season.pkl", "wb") as f:
    pickle.dump(slopes_season, f)
with open("/work/bm1183/m301049/iwp_dists/error_season.pkl", "wb") as f:
    pickle.dump(error_season, f)

# %% plot seasonal slopes
fig, ax = plt.subplots()
for ds in datasets:
    ax.plot(
        hists_season[ds].bin_center,
        slopes_season[ds],
        label=line_labels[ds],
        color=colors[ds],
    )
    ax.fill_between(
        hists_season[ds].bin_center,
        slopes_season[ds] - error_season[ds],
        slopes_season[ds] + error_season[ds],
        color=colors[ds],
        alpha=0.3,
    )
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xscale("log")

# %%
fig, axes = plot_regression(
    t_deseason.sel(time=hists_deseason["ccic"].time),
    hists_deseason["ccic"],
    slopes_monthly["ccic"],
    error_montly["ccic"],
    "CCIC Monthly",
)
fig.savefig("plots/ccic_monthly.png", dpi=300, bbox_inches="tight")
# %%
fig, axes = plot_regression(
    t_deseason.sel(time=hists_deseason["2c"].time),
    hists_deseason["2c"],
    slopes_monthly["2c"],
    error_montly["2c"],
    "2C-ICE Monthly",
)
fig.savefig("plots/2c_monthly.png", dpi=300, bbox_inches="tight")

# %%
fig, axes = plot_regression(
    t_deseason.sel(time=hists_deseason["dardar"].time),
    hists_deseason["dardar"],
    slopes_monthly["dardar"],
    error_montly["dardar"],
    "DARDAR v3.10 Monthly",
)
fig.savefig("plots/dardar_monthly.png", dpi=300, bbox_inches="tight")

# %%
fig, axes = plot_regression(
    t_deseason.sel(time=slice(None, "2025-07")),
    hists_deseason["spare"].sel(time=slice(None, "2025-07")),
    slopes_monthly["spare"],
    error_montly["spare"],
    "SPARE-ICE Monthly",
)
fig.savefig("plots/spare_monthly.png", dpi=300, bbox_inches="tight")

# %% load cre data and hists from icon
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)

experiments = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
}
iwp_hists = {}
for run in ["jed0011", "jed0022", "jed0033"]:
    with open(
        f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/{run}_iwp_hist.pkl",
        "rb",
    ) as f:
        iwp_hists[run] = pickle.load(f)
        iwp_hists[run] = xr.DataArray(
            iwp_hists[run],
            coords={"iwp": cre.iwp},
            dims=["iwp"],
        )


# %% interpolate
iwp_hists_int = {}
for run in ["jed0011", "jed0022", "jed0033"]:
    iwp_hists_int[run] = interpolate_bins(iwp_hists[run], bins, "iwp")

    # check cdf
    print(
        f"{run} sum original: {iwp_hists[run].sel(iwp=slice(iwp_hists_int[run]['bin_center'].min(), None)).sum().item():.3f}, sum interp: {iwp_hists_int[run].sum().item():.3f}"
    )

cre["iwp"] = np.log10(cre["iwp"])
cre = cre.interp(
    iwp=np.log10(hists_monthly["ccic"].bin_center), method="linear"
).drop_vars("iwp")
cre["bin_center"] = 10 ** cre["bin_center"]
iwp_change_icon = {}
temp_deltas = {"jed0022": 4, "jed0033": 2}
for run in ["jed0022", "jed0033"]:
    iwp_change_icon[run] = (
        iwp_hists_int[run] - iwp_hists_int["jed0011"]
    ) / temp_deltas[run]

# %% load rcemip data
ds = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
)
ds["fwp"] = ds["fwp"] * 1e-3
# interpolate histogram
rcemip_pdf = interpolate_bins(ds["f"].mean("model"), bins, "fwp")
diff_rcemip = (rcemip_pdf.sel(SST=305) - rcemip_pdf.sel(SST=295)) / 10

# %% plot slopes
fig, ax = plt.subplots()

for ds in datasets:
    ax.plot(
        hists_monthly[ds].bin_center,
        slopes_monthly[ds],
        label=line_labels[ds],
        color=colors[ds],
    )

for run in ["jed0022", "jed0033"]:
    ax.plot(
        iwp_change_icon[run].bin_center,
        iwp_change_icon[run],
        label=line_labels[run],
        color=colors[run],
        linestyle="--",
    )

ax.plot(
    diff_rcemip.bin_center,
    diff_rcemip,
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xscale("log")

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("dP(I)/dT / K$^{-1}$")
ax.set_xlabel("I / kg m$^{-2}$")
ax.set_xlim(1e-3, 2e1)
ax.legend()
fig.tight_layout()
fig.savefig("plots/slopes_monthly.png", dpi=300, bbox_inches="tight")

# %% plot slopes in %/K
fig, ax = plt.subplots()

for ds in datasets:
    ax.plot(
        hists_monthly[ds].bin_center,
        slopes_monthly[ds] * 100 / hists_monthly[ds].mean("time"),
        label=line_labels[ds],
        color=colors[ds],
    )

for run in ["jed0022", "jed0033"]:
    ax.plot(
        iwp_change_icon[run].bin_center,
        iwp_change_icon[run] * 100 / iwp_hists_int["jed0011"],
        label=line_labels[run],
        color=colors[run],
        linestyle="--",
    )

ax.plot(
    diff_rcemip.bin_center,
    diff_rcemip * 100 / rcemip_pdf.sel(SST=295),
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)

ax.axhline(0, color="k", linewidth=0.5)
ax.set_xscale("log")

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("dP(I)/dT / % K$^{-1}$")
ax.set_xlabel("I / kg m$^{-2}$")
ax.set_xlim(1e-3, 2e1)
ax.set_ylim(-15, 15)
ax.legend()
fig.tight_layout()
fig.savefig("plots/slopes_monthly_perc.png", dpi=300, bbox_inches="tight")

# %% calculate feedback
feedback = {}
for ds in datasets:
    feedback[ds] = slopes_monthly[ds] * cre["net"].values
for run in ["jed0022", "jed0033"]:
    feedback[run] = iwp_change_icon[run] * cre["net"].values

feedback["rcemip"] = diff_rcemip * cre["net"].values


# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[3, 1])
offsets = {
    "jed0033": 0.1,
    "jed0022": 0.2,
    "rcemip": 0.3,
    "ccic": 0.4,
    "2c": 0.5,
    "dardar": 0.6,
    "spare": 0.7,
}

members = datasets + ["rcemip"] + ["jed0022", "jed0033"]
for ds in members:
    axes[0].plot(
        hists_monthly["ccic"].bin_center,
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
fig.savefig("plots/feedback_monthly.png", dpi=300, bbox_inches="tight")


# %%
fig, ax = plt.subplots()

ax.plot(
    hists_monthly["ccic"].bin_center,
    slopes_monthly["ccic"] * 100 / hists_monthly["ccic"].mean("time"),
    label="CCIC",
)
ax.plot(
    hists_monthly["2c"].bin_center,
    slopes_monthly["2c"] * 100 / hists_monthly["2c"].mean("time"),
    label="2C-ICE",
)
ax.plot(
    hists_monthly["dardar"].bin_center,
    slopes_monthly["dardar"] * 100 / hists_monthly["dardar"].mean("time"),
    label="DARDAR v3.10",
)
for run in ["jed0022", "jed0033"]:
    ax.plot(
        iwp_change_icon[run].iwp,
        iwp_change_icon[run] / iwp_hists["jed0011"] * 100,
        label=f"ICON {experiments[run]}",
        linestyle="--",
    )


ax.axhline(0, color="k", linewidth=0.5)
ax.set_xscale("log")

ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(-10, 10)
ax.legend()
# %%
