# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import (
    resample_histograms,
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions, plot_2d_trend
from scipy.signal import detrend
import glob
import re

# %%
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange"}
names = ["ccic", "gpm"]

# %% open gpm
hist_gpm = xr.open_mfdataset("/work/bm1183/m301049/GPM_MERGIR/hists/gpm_*.nc").load()
hist_gpm = hist_gpm.coarsen(bt=2, boundary="trim").sum()

# %% open ccic
files = glob.glob(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d_*.nc"
)
files = [f for f in files if re.search(r"2d_\d{4}\.nc$", f)]
hist_ccic = xr.open_mfdataset(files).load()
hist_ccic = hist_ccic.coarsen(iwp=4, boundary="trim").sum()
# %%
hists = {
    "gpm": hist_gpm,
    "ccic": hist_ccic,
}
# %%
hists_monthly = {}
for name in names:
    hists_monthly[name] = resample_histograms(hists[name])

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %%  detrend and deseasonalize
hists_detrend = {}
temp_detrend = xr.DataArray(detrend(temp), coords=temp.coords, dims=temp.dims)
temp_detrend = deseason(temp_detrend)
for name in names:
    hists_detrend[name] = detrend_hist_2d(hists_monthly[name])
    hists_detrend[name] = deseason(hists_detrend[name])

# %% regression
slopes = {}
p_values = {}

for name in names:
    slopes[name], p_values[name] = regress_hist_temp_2d(
        hists_detrend[name], temp_detrend
    )
    slopes[name] = (
        slopes[name] * 100 / hists_monthly[name].mean("time")
    )  # convert to %/K

# %% plot slopes ccic
mean_hists = {}
mean_hists["ccic"] = hists["ccic"]["hist"].sel(iwp=slice(1e-1, 10)).sum("time") / hists[
    "ccic"
]["size"].sum("time")
fig, axes = plot_2d_trend(
    mean_hists["ccic"] / mean_hists["ccic"].max(),
    slopes["ccic"].sel(iwp=slice(1e-1, 10)),
    p_values["ccic"].sel(iwp=slice(1e-1, 10)),
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/ccic_2d_trend.png", dpi=300)

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
fig.savefig("plots/diurnal_cycle/gpm_2d_trend.png", dpi=300)

# %% plot mean diurnal cycle
mean_ccic = (
    hists["ccic"].sel(iwp=slice(1e-1, 10)).sum(["time", "iwp"])["hist"]
    / hists["ccic"].sel(iwp=slice(1e-1, 10))["hist"].sum()
)
mean_gpm = (
    hists["gpm"].sel(bt=slice(190, 230)).sum(["time", "bt"])["hist"]
    / hists["gpm"].sel(bt=slice(190, 230))["hist"].sum()
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mean_ccic["local_time"], mean_ccic, label="CCIC", color=color["ccic"])
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

# %% plot cumulative area change
fig, ax = plt.subplots(figsize=(8, 5))

# %%
