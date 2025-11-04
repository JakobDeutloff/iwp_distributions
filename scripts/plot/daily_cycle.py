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

# %% open CCIC
path = "/work/bm1183/m301049/ccic_daily_cycle/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_daily_cycle_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hists_ccic = xr.concat(hist_list, dim="time")

# %% 
hists_ccic_monthly = hists_ccic.resample(time="1ME").sum()
hists_ccic_monthly['time'] = pd.to_datetime(hists_ccic_monthly['time'].dt.strftime('%Y-%m'))
hists_ccic_monthly = hists_ccic_monthly['hist'] / hists_ccic_monthly['size']
hists_ccic_monthly = hists_ccic_monthly.transpose('local_time', 'time')

# %% load era5 surface temp
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

# %%  detrend and deseasonalize 

# temperature
t_detrend = xr.DataArray(detrend(t_month), coords=t_month.coords, dims=t_month.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)
t_smooth = t_deseason.rolling(time=1, center=True).mean()
t_smooth["time"] = pd.to_datetime(t_smooth["time"].dt.strftime("%Y-%m"))

# histograms ccic
hists_detrend = nan_detrend(hists_ccic_monthly)
hists_deseason = hists_detrend.groupby("time.month") - hists_detrend.groupby(
    "time.month"
).mean("time")
hists_deseason["time"] = pd.to_datetime(hists_deseason["time"].dt.strftime("%Y-%m"))
hists_smooth_ccic = (
    hists_deseason.rolling(time=1, center=True).mean().isel(time=slice(1, -1))
)
hists_smooth_ccic["time"] = pd.to_datetime(
    hists_smooth_ccic["time"].dt.strftime("%Y-%m")
)

# %% regression 

slopes_ccic = []
err_ccic = []
hists_dummy = hists_smooth_ccic.where(hists_deseason.notnull(), drop=True)
temp_vals_ccic = t_smooth.sel(time=hists_dummy.time).values
for i in range(hists_dummy.local_time.size):
    hist_vals = hists_dummy.isel(local_time=i).values
    slope, intercept, r_value, p_value, std_err = linregress(
        temp_vals_ccic, hist_vals
    )
    slopes_ccic.append(slope)
    err_ccic.append(std_err)

slopes_ccic = xr.DataArray(
    slopes_ccic,
    coords={"local_time": hists_dummy.local_time},
    dims=["local_time"],
)
err_ccic = xr.DataArray(
    err_ccic,
    coords={"local_time": hists_dummy.local_time},
    dims=["local_time"],
)

# %% load icon 
runs = ['jed0011', 'jed0022', 'jed0033']
temp_delta = {
    'jed0011': 0,
    'jed0022': 4,
    'jed0033': 2,
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
    change_icon[run] = (
        hists_icon[run] - hists_icon['jed0011']
    ) / temp_delta[run]

# %% plot of mean daily cycle
mean_ccic = hists_ccic.mean('time')
fig, ax = plt.subplots(figsize=(8, 5))
ax.stairs(
    mean_ccic['hist']/mean_ccic['size'], np.arange(0, 25, 1), label=line_labels['ccic'], color=colors['ccic'], linewidth=2)
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

# %% plot cahnge in diurnal cycle 
fig, ax = plt.subplots(figsize=(8, 5))
mean_ccic = hists_ccic.mean('time')['hist']/hists_ccic.mean('time')['size']
# ax.plot(slopes_ccic["local_time"], slopes_ccic*100/mean_ccic, label="CCIC", color=colors['ccic'], linewidth=2)

# ax.fill_between(
#     slopes_ccic["local_time"],
#     slopes_ccic*100/mean_ccic - err_ccic*100/mean_ccic,
#     slopes_ccic*100/mean_ccic + err_ccic*100/mean_ccic,
#     alpha=0.3,
#     color=colors['ccic']
# )
for run in runs[1:]:
    ax.plot(
        slopes_ccic["local_time"],
        change_icon[run]*100/hists_icon['jed0011'],
        label=line_labels[run],
        color=colors[run],
    )
ax.set_ylabel("dP(I $>$ 1 kg m$^{-2}$)/dT / % K$^{-1}$")
ax.set_xlabel("Local Time / h")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim([0, 23.9])
ax.set_ylim([-4.1, 3])
ax.legend()
fig.tight_layout()
fig.savefig("plots/daily_cycle_change_icon.png", dpi=300, bbox_inches="tight")

# %%
