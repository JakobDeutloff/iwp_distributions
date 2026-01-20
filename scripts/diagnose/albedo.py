# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

# %% load icon data
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

# %% load iwp bt fit coeffs
with open('/work/bm1183/m301049/diurnal_cycle_dists/bt_iwp_fig_coeffs.pkl', 'rb') as f:
    bt_iwp_coeffs = pickle.load(f)

# %% load bt_of_iwp
bt_of_iwp = xr.open_dataset(
    '/work/bm1183/m301049/diurnal_cycle_dists/bt_of_iwp.nc'
).mean('time')
iwp_of_bt = xr.DataArray(
    np.log10(bt_of_iwp['iwp'].values),
    coords={'bt': bt_of_iwp['bt_of_iwp'].values},
    dims=['bt'])
iwp_of_bt = iwp_of_bt.isel(bt=slice(45, None))

# %% functions
def calc_hc_albedo(a_cs, a_as):
    return (a_as - a_cs) / (a_cs * (a_as - 2) + 1)

def iwp_from_bt(bt):
    """
    Calculate iwp from BT by using inverse of linear fit from bt_iwp.py
    """
    slope = bt_iwp_coeffs['slope']
    intercept = bt_iwp_coeffs['intercept']
    iwp = 10**((bt - intercept) / slope)
    return iwp

def iwp_from_bt_corr(bt):
    return 10 ** iwp_of_bt.interp(bt=bt).values


# %% initialize datasets
sw_vars = xr.Dataset()
mean_sw_vars = pd.DataFrame()
iwp_bins = np.logspace(-3, 2, 254)[::4]
bt_bins = np.arange(150, 340, 1)
bt_bins = np.insert(bt_bins, 0, 0)
bt_bins = bt_bins[::2]
iwp_bins_bt = iwp_from_bt_corr(bt_bins)
time_bins = np.linspace(0, 24, 25)
time_points = (time_bins[1:] + time_bins[:-1]) / 2
binned_hc_albedo_iwp = np.zeros([len(iwp_bins) - 1, len(time_bins) - 1]) * np.nan
binned_hc_albedo_bt = np.zeros([len(bt_bins) - 1, len(time_bins) - 1]) * np.nan

# %% set mask
mask_parameterisation = (ds['mask_low_cloud'] == 0)

# %% calculate high cloud albedo
sw_vars["wetsky_albedo"] = np.abs(ds["rsutws"] / ds["rsdt"])
sw_vars["allsky_albedo"] = np.abs(ds["rsut"] / ds["rsdt"])
sw_vars["clearsky_albedo"] = np.abs(ds["rsutcs"] / ds["rsdt"])
cs_albedo = xr.where(
    ds["conn"], sw_vars["clearsky_albedo"], sw_vars["wetsky_albedo"]
)
sw_vars["high_cloud_albedo"] = calc_hc_albedo(cs_albedo, sw_vars["allsky_albedo"])

# %% calculate mean albedos by weighting with the incoming SW radiation in IWP bins

for i in range(len(iwp_bins) - 1):
    IWP_mask = (ds["iwp"] > iwp_bins[i]) & (ds["iwp"] <= iwp_bins[i + 1])
    for j in range(len(time_bins) - 1):
        time_mask = (ds['time_local'] > time_bins[j]) & (
            ds['time_local'] <= time_bins[j + 1]
        )
        binned_hc_albedo_iwp[i, j] = float(
            sw_vars["high_cloud_albedo"]
            .where(IWP_mask & time_mask & mask_parameterisation)
            .mean()
            .values
        )
# %%calculate albedo for bt bins 
for i in range(len(iwp_bins_bt) - 1):
    IWP_mask = (ds["iwp"] <= iwp_bins_bt[i]) & (ds["iwp"] > iwp_bins_bt[i + 1])
    for j in range(len(time_bins) - 1):
        time_mask = (ds['time_local'] > time_bins[j]) & (
            ds['time_local'] <= time_bins[j + 1]
        )
        binned_hc_albedo_bt[i, j] = float(
            sw_vars["high_cloud_albedo"]
            .where(IWP_mask & time_mask & mask_parameterisation)
            .mean()
            .values
        )

# %% save albedo
albedo_iwp = xr.Dataset(
    {
        "hc_albedo": (("iwp", "local_time"), binned_hc_albedo_iwp),
    },
    coords={
        "iwp": hist_ccic['iwp'],
        "local_time": time_points,
    },
)
albedo_iwp.to_netcdf('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc')

# %%
albedo_bt = xr.Dataset(
    {
        "hc_albedo": (("bt", "local_time"), binned_hc_albedo_bt),
    },
    coords={
        "bt": hist_gpm['bt'],
        "local_time": time_points,
    },
)
albedo_bt.to_netcdf('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc')

# %%
