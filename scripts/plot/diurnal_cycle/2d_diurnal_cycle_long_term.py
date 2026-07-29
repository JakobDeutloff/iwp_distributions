# %%
import xarray as xr
from src.helper_functions import (
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions, plot_2d_trend, plot_2d_trend_talk
from scipy.signal import detrend
import matplotlib.pyplot as plt


# %% load ccic and gpm data
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange", "icon": "green", "era5": "blue"}
names = ["ccic", "gpm", 'era5']
dim = {"ccic": "iwp", "gpm": "bt", "icon": "iwp", "era5": "iwp"}

hists = {}
hists["ccic"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
)
hists["gpm"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all.nc"
)
hists["era5"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_all.nc"
)

# %% load albedo
albedo_iwp = xr.open_dataset("/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc")
albedo_bt = xr.open_dataset("/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc")
SW_in = xr.open_dataarray(
    "/work/bm1183/m301049/icon_hcap_data/publication/incoming_sw/SW_in_daily_cycle.nc"
)
SW_in = SW_in.interp(time_points=hists["ccic"]["local_time"], method="linear")

# %% calculate cloud fraction
cf = {}
for name in names:
    cf[name] = hists[name]['hist'] / hists[name]['size']
# %% normalise cloud fraction
cf_norm = {}
for name in names:
    cf_norm[name] = cf[name] / cf[name].sum('local_time')

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %% calc annual means
temp_ann = temp.groupby("time.year").mean("time").rename(year="time")
cf_ann = {}
cf_norm_ann = {}
for name in names:
    cf_ann[name] = cf[name].groupby("time.year").mean("time").rename(year="time")
    cf_norm_ann[name] = cf_ann[name] / cf_ann[name].sum('local_time')

# %% regression
import numpy as np
from scipy.stats import linregress

def regress_hist_temp_2d_trend(cf, temp):
    if "bt" in cf.dims:
        detrend_dim = "bt"
    else:
        detrend_dim = "iwp"

    slopes = xr.zeros_like(cf.isel(time=0))
    p_values = xr.zeros_like(cf.isel(time=0))
    slope_temp, _, _, _, _ = linregress(
        np.arange(len(temp.sel(time=cf.time).values)), temp.sel(time=cf.time).values
    )
    for i in cf.local_time:
        for j in cf[detrend_dim]:
            cf_vals = cf.sel({"local_time": i, detrend_dim: j})
            cf_vals = cf_vals.where(np.isfinite(cf_vals), drop=True)
            slope_freq, _, _, p_value, _ = linregress(
                np.arange(len(cf_vals.values)), cf_vals.values
            )
            slopes.loc[{"local_time": i, detrend_dim: j}] = slope_freq/slope_temp
            p_values.loc[{"local_time": i, detrend_dim: j}] = p_value
    
    slopes_perc = slopes * 100 / cf.mean("time")  # convert to % / K
    return slopes_perc, p_values

slopes = {}
p_values = {}

for name in names:
    slopes[name], p_values[name] = regress_hist_temp_2d_trend(
        cf_norm[name].fillna(0), temp
    )

# %% calculate feedback
cf_change = {}
feedbacks = {}
feedbacks_int = {}

cutoffs = {
    "ccic": {"iwp": slice(1e-1, None)},
    "gpm": {"bt": slice(None, 260)},
    "era5": {"iwp": slice(1e-1, None)},
}
albedo = {
    "ccic": albedo_iwp["hc_albedo"],
    "gpm": albedo_bt["hc_albedo"],
    "era5": albedo_iwp["hc_albedo"],
}

for name in names:
    cf_change[name] = (slopes[name] / 100) *  cf[name].mean('time')  # 1/K
    feedbacks[name] = -1 * (
        (cf_change[name] * SW_in * albedo[name].values.T)
        - ((cf_change[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int[name] = feedbacks[name].sel(cutoffs[name]).sum()  # W / m^2 / K


# %% calculate cumulative feedback 
feedback_cum = {}
feedback_cum_ann = {}
feedback_cum_reg = {}
feedback_cum_reg_ann = {}
feedback_cum["ccic"] = (
    feedbacks["ccic"]
    .sel(cutoffs["ccic"])
    .sum("local_time")
    .cumsum("iwp")
)
feedback_cum["gpm"] = (
    feedbacks["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
)
feedback_cum["era5"] = (
    feedbacks["era5"]
    .sel(cutoffs["era5"])
    .sum("local_time")
    .cumsum("iwp")
)

# %% plot slopes ccic trend
fig, axes = plot_2d_trend(
    cf["ccic"].mean('time'),
    slopes["ccic"],
    cf_change["ccic"],
    feedbacks["ccic"],
    p_values["ccic"],
    feedback_cum["ccic"],
    err_cum=0,
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/long_term/trend_2d_ccic.pdf", bbox_inches="tight")

# %% plot slopes gpm trend
fig, axes = plot_2d_trend(
    cf["gpm"].mean('time'),
    slopes["gpm"],
    cf_change["gpm"],
    feedbacks["gpm"],
    p_values["gpm"],
    feedback_cum["gpm"],
    err_cum=0,
    dim="bt",
)
fig.savefig("plots/diurnal_cycle/long_term/trend_2d_gpm.pdf", bbox_inches="tight")
# %% plot slopes era5 trend
fig, axes = plot_2d_trend(
    cf["era5"].mean('time'),
    slopes["era5"],
    cf_change["era5"],
    feedbacks["era5"],
    p_values["era5"],
    feedback_cum["era5"],
    err_cum=0,
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/long_term/trend_2d_era5.pdf", bbox_inches="tight")

# %%
