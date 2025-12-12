# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %%
run = 'jed0011'
exp_name = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}
colors = {'jed0011': 'k', 'jed0022': 'r', 'jed0033': 'orange'}
ds = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/{exp_name[run]}/production/random_sample/{run}_randsample_processed_64.nc"
).sel(index=slice(None, 1e6))

# %% load hists for bins 
hist_ccic = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
)
hist_gpm = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all.nc"
)

# %% initialize datasets
sw_vars = xr.Dataset()
mean_sw_vars = pd.DataFrame()

# %% set mask
mask_parameterisation = (ds['mask_low_cloud'] == 0)

# %% calculate high cloud albedo
def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as - 2) + 1)


sw_vars["wetsky_albedo"] = np.abs(ds["rsutws"] / ds["rsdt"])
sw_vars["allsky_albedo"] = np.abs(ds["rsut"] / ds["rsdt"])
sw_vars["clearsky_albedo"] = np.abs(ds["rsutcs"] / ds["rsdt"])
cs_albedo = xr.where(
    ds["conn"], sw_vars["clearsky_albedo"], sw_vars["wetsky_albedo"]
)
sw_vars["high_cloud_albedo"] = calc_hc_albedo(cs_albedo, sw_vars["allsky_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins
IWP_bins = np.logspace(-3, 2, 254)[::4]
bt_bins = np.arange(175, 330, 1)[::2]
time_bins = np.linspace(0, 24, 25)
time_points = (time_bins[1:] + time_bins[:-1]) / 2
binned_hc_albedo = np.zeros([len(IWP_bins) - 1, len(time_bins) - 1]) * np.nan

for i in range(len(IWP_bins) - 1):
    IWP_mask = (ds["iwp"] > IWP_bins[i]) & (ds["iwp"] <= IWP_bins[i + 1])
    for j in range(len(time_bins) - 1):
        time_mask = (ds['time_local'] > time_bins[j]) & (
            ds['time_local'] <= time_bins[j + 1]
        )
        binned_hc_albedo[i, j] = float(
            sw_vars["high_cloud_albedo"]
            .where(IWP_mask & time_mask & mask_parameterisation)
            .mean()
            .values
        )

# %% plot albedo in iwp bins
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pcol = ax.pcolormesh(IWP_bins, time_bins, binned_hc_albedo.T, cmap="viridis")
ax.set_ylabel("Local time")
ax.set_xscale("log")
ax.set_xlabel("IWP / kg m$^{-2}$")
ax.set_xlim([1e-4, 1e1])
fig.colorbar(pcol, label="High Cloud Albedo", location="bottom", ax=ax, shrink=0.8)

# %% save albedo in iwp bins
albedo = xr.Dataset(
    {
        "hc_albedo": (("iwp", "local_time"), binned_hc_albedo),
    },
    coords={
        "iwp": hist_ccic['iwp'],
        "local_time": time_points,
    },
)
albedo.to_netcdf('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo.nc')

# %%
