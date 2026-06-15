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
color = {"ccic": "black", "gpm": "orange", "icon": "green"}
names = ["ccic", "gpm"]
dim = {"ccic": "iwp", "gpm": "bt", "icon": "iwp"}

hists = {}
hists["ccic"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
)
hists["gpm"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all.nc"
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
for name in ['ccic', 'gpm']:
    cf[name] = hists[name]['hist'] / hists[name]['hist'].sum(['local_time', dim[name]])
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

# %% trend analysis
slopes = {}
p_values = {}

for name in names:
    slopes[name], p_values[name] = regress_hist_temp_2d_trend(
        cf_norm[name].fillna(0), temp
    )

# %% trend analysis on annual means
slopes_ann = {}
p_values_ann = {}
for name in names:
    slopes_ann[name], p_values_ann[name] = regress_hist_temp_2d_trend(
        cf_norm_ann[name].fillna(0), temp_ann
    )

# %% normal regression 
slopes_reg = {}
p_values_reg = {}
for name in names:
    slopes_reg[name], p_values_reg[name] = regress_hist_temp_2d(
        cf_norm[name].fillna(0), temp, cf_norm[name]
    )

# %% normal regression on annual means
slopes_reg_ann = {}
p_values_reg_ann = {}
for name in names:
    slopes_reg_ann[name], p_values_reg_ann[name] = regress_hist_temp_2d(
        cf_norm_ann[name].fillna(0), temp_ann, cf_norm_ann[name]
    )




# %% calculate feedback
cf_change = {}
feedbacks = {}
feedbacks_int = {}

cf_change_ann = {}
feedbacks_ann = {}
feedbacks_int_ann = {}

cf_change_reg = {}
feedbacks_reg = {}
feedbacks_int_reg = {}

cf_change_reg_ann = {}
feedbacks_reg_ann = {}
feedbacks_int_reg_ann = {}

cutoffs = {
    "ccic": {"iwp": slice(1e-1, None)},
    "gpm": {"bt": slice(None, 260)},
}
albedo = {
    "ccic": albedo_iwp["hc_albedo"],
    "gpm": albedo_bt["hc_albedo"],
    "icon": albedo_iwp["hc_albedo"],
}

for name in ['ccic', 'gpm']:
    cf_change[name] = (slopes[name] / 100) *  cf[name].mean('time')  # 1/K
    feedbacks[name] = -1 * (
        (cf_change[name] * SW_in * albedo[name].values.T)
        - ((cf_change[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int[name] = feedbacks[name].sel(cutoffs[name]).sum()  # W / m^2 / K

    cf_change_ann[name] = (slopes_ann[name] / 100) *  cf_ann[name].mean('time')  # 1/K
    feedbacks_ann[name] = -1 * (
        (cf_change_ann[name] * SW_in * albedo[name].values.T)
        - ((cf_change_ann[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int_ann[name] = feedbacks_ann[name].sel(cutoffs[name]).sum()  # W / m^2 / K   
    cf_change_reg[name] = (slopes_reg[name] / 100) *  cf[name].mean('time')  # 1/K
    feedbacks_reg[name] = -1 * (
        (cf_change_reg[name] * SW_in * albedo[name].values.T)
        - ((cf_change_reg[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int_reg[name] = feedbacks_reg[name].sel(cutoffs[name]).sum()  # W / m^2 / K
    cf_change_reg_ann[name] = (slopes_reg_ann[name] / 100) *  cf_ann[name].mean('time')  # 1/K
    feedbacks_reg_ann[name] = -1 * (
        (cf_change_reg_ann[name] * SW_in * albedo[name].values.T)
        - ((cf_change_reg_ann[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int_reg_ann[name] = feedbacks_reg_ann[name].sel(cutoffs[name]).sum()  # W / m^2 / K



# %% calculate cumulative feedback from bootstrapped samples
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
feedback_cum_ann["ccic"] = (
    feedbacks_ann["ccic"]
    .sel(cutoffs["ccic"])
    .sum("local_time")
    .cumsum("iwp")
)
feedback_cum_ann["gpm"] = (
    feedbacks_ann["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
)
feedback_cum_reg["ccic"] = (
    feedbacks_reg["ccic"]
    .sel(cutoffs["ccic"])      
    .sum("local_time")
    .cumsum("iwp")
)
feedback_cum_reg["gpm"] = (
    feedbacks_reg["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
)
feedback_cum_reg_ann["ccic"] = (
    feedbacks_reg_ann["ccic"]
    .sel(cutoffs["ccic"])
    .sum("local_time")
    .cumsum("iwp")
)
feedback_cum_reg_ann["gpm"] = (
    feedbacks_reg_ann["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
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
# %% plot slopes ccic trend annual means
fig, axes = plot_2d_trend(
    cf_ann["ccic"].mean('time'),
    slopes_ann["ccic"],
    cf_change_ann["ccic"],
    feedbacks_ann["ccic"],
    p_values_ann["ccic"],
    feedback_cum_ann["ccic"],
    err_cum=0,
    dim="iwp",
)

# %% plot slopes gpm trend annual means
fig, axes = plot_2d_trend(
    cf_ann["gpm"].mean('time'),
    slopes_ann["gpm"],
    cf_change_ann["gpm"],
    feedbacks_ann["gpm"],
    p_values_ann["gpm"],
    feedback_cum_ann["gpm"],
    err_cum=0,
    dim="bt",
)

# %% plot slopes ccic trend normal regression
fig, axes = plot_2d_trend(
    cf["ccic"].mean('time'),
    slopes_reg["ccic"],
    cf_change_reg["ccic"],
    feedbacks_reg["ccic"],
    p_values_reg["ccic"],
    feedback_cum_reg["ccic"],
    err_cum=0,
    dim="iwp",
)

# %% plot slopes gpm trend normal regression
fig, axes = plot_2d_trend(
    cf["gpm"].mean('time'),
    slopes_reg["gpm"],
    cf_change_reg["gpm"],
    feedbacks_reg["gpm"],
    p_values_reg["gpm"],
    feedback_cum_reg["gpm"],
    err_cum=0,
    dim="bt",
)

# %% plot slopes ccic trend normal regression annual means
fig, axes = plot_2d_trend(
    cf_ann["ccic"].mean('time'),
    slopes_reg_ann["ccic"],
    cf_change_reg_ann["ccic"],
    feedbacks_reg_ann["ccic"],
    p_values_reg_ann["ccic"],
    feedback_cum_reg_ann["ccic"],
    err_cum=0,
    dim="iwp",
)

# %% plot slopes gpm trend normal regression annual means
fig, axes = plot_2d_trend(
    cf_ann["gpm"].mean('time'),
    slopes_reg_ann["gpm"],
    cf_change_reg_ann["gpm"],
    feedbacks_reg_ann["gpm"],
    p_values_reg_ann["gpm"],
    feedback_cum_reg_ann["gpm"],
    err_cum=0,
    dim="bt",
)

# %%
