# %%
import xarray as xr
import pandas as pd
import glob
import re

# %% load ccic data
names = ["all", "sea", "land"]
hists_ccic = {}
hists_gpm = {}
files = glob.glob(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d*.nc"
)
files_all = [f for f in files if re.search(r"2d_all_\d{4}\.nc$", f)]
files_sea = [f for f in files if re.search(r"2d_sea_\d{4}\.nc$", f)]
hists_ccic["all"] = xr.open_mfdataset(files_all).load()
hists_ccic["sea"] = xr.open_mfdataset(files_sea).load()
hists_ccic["land"] = hists_ccic["all"] - hists_ccic["sea"]
# %% load gpm data
hists_gpm["all"] = xr.open_mfdataset(
    "/work/bm1183/m301049/GPM_MERGIR/hists/gpm_extratropics_2d_hist_all*.nc"
).load()
hists_gpm["sea"] = xr.open_mfdataset(
    "/work/bm1183/m301049/GPM_MERGIR/hists/gpm_extratropics_2d_hist_sea*.nc"
).load()
hists_gpm["land"] = hists_gpm["all"] - hists_gpm["sea"]
# %% coarsen hists 
hists_ccic_coarse = {}
hists_gpm_coarse = {}
for name in names:
    hists_ccic_coarse[name] = hists_ccic[name].coarsen(iwp=4, boundary="trim").sum()
    hists_gpm_coarse[name] = hists_gpm[name].coarsen(bt=2, boundary="trim").sum()

# %% resample histograms to monthly
hists_ccic_monthly = {}
hists_gpm_monthly = {}
for name in names:
    hists_ccic_monthly[name] = hists_ccic_coarse[name].resample(time="1ME").sum()
    hists_ccic_monthly[name]['time'] = pd.to_datetime(hists_ccic_monthly[name]["time"].dt.strftime("%Y-%m"))
    hists_gpm_monthly[name] = hists_gpm_coarse[name].resample(time="1ME").sum()
    hists_gpm_monthly[name]['time'] = pd.to_datetime(hists_gpm_monthly[name]["time"].dt.strftime("%Y-%m"))

# %% save processed data
for name in names:
    # hists_ccic_monthly[name].to_netcdf(
    #     f"/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_{name}.nc"
    # )
    hists_gpm_monthly[name].to_netcdf(
        f"/work/bm1183/m301049/diurnal_cycle_dists/gpm_extratropics_2d_monthly_{name}.nc"
    )

# %%
