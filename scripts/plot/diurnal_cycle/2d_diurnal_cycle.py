# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import (
    normalise_histograms,
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions, plot_2d_trend
from scipy.signal import detrend


# %% load ccic and gpm data
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange", "icon": "green"}
names = ["ccic", "gpm"]

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
# %% load bootstrapped feedbacks
feedbacks_bs = {
    "ccic": xr.open_dataarray(
        "/work/bm1183/m301049/diurnal_cycle_dists/ccic_bootstrap_feedback_2d.nc"
    ),
    "gpm": xr.open_dataarray(
        "/work/bm1183/m301049/diurnal_cycle_dists/gpm_bootstrap_feedback_2d.nc"
    ),
}

# %% open icon
hist_icon_control = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon_hcap_data/control/production/daily_cycle_hist_2d.nc"
    )
    .coarsen(iwp=4, boundary="trim")
    .sum()
)
hist_icon_4k = (
    xr.open_dataset(
        "/work/bm1183/m301049/icon_hcap_data/plus4K/production/daily_cycle_hist_2d.nc"
    )
    .coarsen(iwp=4, boundary="trim")
    .sum()
)
hists["icon"] = hist_icon_control
# %%
hists_monthly = {}
for name in names:
    hists_monthly[name] = normalise_histograms(hists[name])

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
        hists_detrend[name], temp_detrend, hists_monthly[name]
    )
# %% calculate slopes icon
hist_icon_control_norm = hist_icon_control["hist"].sum("time") / hist_icon_control[
    "hist"
].sum(["time", "local_time"])
hist_icon_4k_norm = hist_icon_4k["hist"].sum("time") / hist_icon_4k["hist"].sum(
    ["time", "local_time"]
)
slopes["icon"] = ((hist_icon_4k_norm - hist_icon_control_norm) * 100) / (
    4.0 * hist_icon_control_norm
)  # % / K

# %% calculate feedback
area_fraction = {}
area_change = {}
feedbacks = {}
feedbacks_int = {}
cutoffs = {
    "ccic": {"iwp": slice(1e-1, None)},
    "gpm": {"bt": slice(None, 260)},
    "icon": {"iwp": slice(1e-1, None)},
}
albedo = {
    "ccic": albedo_iwp["hc_albedo"],
    "gpm": albedo_bt["hc_albedo"],
    "icon": albedo_iwp["hc_albedo"],
}

for name in ['ccic', 'gpm', 'icon']:
    area_fraction[name] = hists[name]["hist"].sum("time") / hists[name]["size"].sum(
        "time"
    )  # 1/1
    area_change[name] = (slopes[name] / 100) * area_fraction[name]  # 1/K
    feedbacks[name] = -1 * (
        (area_change[name] * SW_in * albedo[name].values.T)
        - ((area_change[name]) * SW_in * 0.1)
    )  # W / m^2 / K
    feedbacks_int[name] = feedbacks[name].sel(cutoffs[name]).sum()  # W / m^2 / K


# %% calculate cumulative feedback from bootstrapped samples
err_feedback_bs = {}
feedback_cum_bs = {}
feedback_cum_bs["ccic"] = (
    feedbacks_bs["ccic"]
    .sel(cutoffs["ccic"])
    .sum("local_time")
    .cumsum("iwp")
    .mean(dim="iteration")
)
err_feedback_bs["ccic"] = (
    feedbacks_bs["ccic"]
    .sel(cutoffs["ccic"])
    .sum("local_time")
    .cumsum("iwp")
    .std(dim="iteration")
)
feedback_cum_bs["gpm"] = (
    feedbacks_bs["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
    .mean(dim="iteration")
)
err_feedback_bs["gpm"] = (
    feedbacks_bs["gpm"]
    .sel(cutoffs["gpm"])
    .sum("local_time")
    .isel(bt=slice(None, None, -1))
    .cumsum("bt")
    .std(dim="iteration")
)
feedback_cum_bs["icon"] = (
    feedbacks["icon"].sel(cutoffs["icon"]).sum("local_time").cumsum("iwp")
)
err_feedback_bs["icon"] = xr.zeros_like(feedback_cum_bs["icon"])
# %% plot slopes ccic
fig, axes = plot_2d_trend(
    area_fraction["ccic"],
    slopes["ccic"],
    area_change["ccic"],
    feedbacks["ccic"],
    p_values["ccic"],
    feedback_cum_bs["ccic"],
    err_feedback_bs["ccic"],
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/publication/ccic_2d_trend.pdf", bbox_inches='tight')

# %% plot slopes gpm
fig, axes = plot_2d_trend(
    area_fraction["gpm"],
    slopes["gpm"],
    area_change["gpm"],
    feedbacks["gpm"],
    p_values["gpm"],
    feedback_cum_bs["gpm"],
    err_feedback_bs["gpm"],
    dim="bt",
)
fig.savefig("plots/diurnal_cycle/publication/gpm_2d_trend.pdf", bbox_inches='tight')

# %% plot slopes icon
# 1 / K
fig, axes = plot_2d_trend(
    area_fraction["icon"],
    slopes["icon"].sel(iwp=slice(1e-1, 10)),
    area_change["icon"],
    feedbacks["icon"],
    xr.full_like(slopes["icon"], 0),
    feedback_cum_bs["icon"],
    err_feedback_bs["icon"],
    dim="iwp",
)
fig.savefig("plots/diurnal_cycle/icon_2d_trend.png", dpi=300)

# %%
