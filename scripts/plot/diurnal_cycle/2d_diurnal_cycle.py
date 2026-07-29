# %%
import xarray as xr
from src.helper_functions import (
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions, plot_2d_trend, plot_2d_trend_talk
from scipy.signal import detrend

# %% load ccic and gpm data
colors, line_labels, linestyles = definitions()
color = {"ccic": "black", "gpm": "orange", "icon": "green"}
names = ["ccic", "gpm", "era5"]
dim = {"ccic": "iwp", "gpm": "bt", "icon": "iwp", "era5": "iwp"}

hists = {}
hists["ccic"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all_weighted.nc"
)
hists["gpm"] = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all_weighted.nc"
)
hists["era5"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_all_weighted.nc"
)

# %% load albedo
albedo_iwp = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc"
)
albedo_bt = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc"
)
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
        "/work/bm1183/m301049/icon_hcap_data/plus2K/production/daily_cycle_hist_2d.nc"
    )
    .coarsen(iwp=4, boundary="trim")
    .sum()
)
hists["icon"] = hist_icon_control
# %% calculate cloud fraction
cf = {}
for name in ["ccic", "gpm", "era5"]:
    cf[name] = hists[name]["hist"] / hists[name]["size"]
cf["icon"] = hists["icon"]["hist"] / hists["icon"]["size"]
# %% normalise cloud fraction
cf_norm = {}
for name in names:
    cf_norm[name] = cf[name] / cf[name].sum("local_time")

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %%  detrend and deseasonalize
cf_detrend = {}
temp_detrend = xr.DataArray(detrend(temp), coords=temp.coords, dims=temp.dims)
temp_detrend = deseason(temp_detrend)
for name in names:
    cf_detrend[name] = detrend_hist_2d(cf_norm[name])
    cf_detrend[name] = deseason(cf_detrend[name])

# %% regression
slopes = {}
p_values = {}

for name in names:
    slopes[name], p_values[name] = regress_hist_temp_2d(
        cf_detrend[name], temp_detrend, cf_norm[name]
    )
# %% calculate slopes icon
hist_icon_control_norm = hist_icon_control["hist"].sum("time") / hist_icon_control[
    "hist"
].sum(["time", "local_time"])
hist_icon_4k_norm = hist_icon_4k["hist"].sum("time") / hist_icon_4k["hist"].sum(
    ["time", "local_time"]
)
slopes["icon"] = ((hist_icon_4k_norm - hist_icon_control_norm) * 100) / (
    2.0 * hist_icon_control_norm
)  # % / K

# %% calculate feedback
cf_change = {}
feedbacks = {}
feedbacks_int = {}
cutoffs = {
    "ccic": {"iwp": slice(1e-1, None)},
    "gpm": {"bt": slice(None, 260)},
    "icon": {"iwp": slice(1e-1, None)},
    "era5": {"iwp": slice(1e-1, None)},
}
albedo = {
    "ccic": albedo_iwp["hc_albedo"],
    "gpm": albedo_bt["hc_albedo"],
    "icon": albedo_iwp["hc_albedo"],
    "era5": albedo_iwp["hc_albedo"],
}

for name in ["ccic", "gpm", "icon", "era5"]:
    cf_change[name] = (slopes[name] / 100) * cf[name].mean("time")  # 1/K
    feedbacks[name] = -1 * (
        (cf_change[name] * SW_in * albedo[name].values.T)
        - ((cf_change[name]) * SW_in * 0.13)
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
    cf["ccic"].mean("time"),
    slopes["ccic"] - slopes["ccic"].mean("local_time"),
    cf_change["ccic"],
    feedbacks["ccic"],
    p_values["ccic"],
    feedback_cum_bs["ccic"],
    err_feedback_bs["ccic"],
    dim="iwp",
)
# fig.savefig("plots/diurnal_cycle/publication/ccic_2d_trend.pdf", bbox_inches='tight')

# %% numbers for feedback
mean_feedback = feedback_cum_bs["ccic"].isel(iwp=-1).values
err_feedback = err_feedback_bs["ccic"].isel(iwp=-1).values
print(f"Feedback: {mean_feedback:.2f} +/- {err_feedback:.2f} W/m^2/K")

# %% plot slopes era5
fig, axes = plot_2d_trend(
    cf["era5"].mean("time"),
    slopes["era5"] - slopes["era5"].mean("local_time"),
    cf_change["era5"],
    feedbacks["era5"],
    p_values["era5"],
    feedbacks["era5"].sel(cutoffs["era5"]).sum("local_time").cumsum("iwp"),
    xr.full_like(
        feedbacks["era5"].sel(cutoffs["era5"]).sum("local_time").cumsum("iwp"), 0
    ),
    dim="iwp",
)
#fig.savefig("plots/diurnal_cycle/long_term/int_var_2d_era5.pdf", bbox_inches="tight")

# %% plot slopes gpm
fig, axes = plot_2d_trend(
    cf["gpm"].mean("time"),
    slopes["gpm"],
    cf_change["gpm"],
    feedbacks["gpm"],
    p_values["gpm"],
    feedback_cum_bs["gpm"],
    err_feedback_bs["gpm"],
    dim="bt",
)
#fig.savefig("plots/diurnal_cycle/publication/gpm_2d_trend.pdf", bbox_inches="tight")

# %% plot slopes icon
# 1 / K
fig, axes = plot_2d_trend(
    cf["icon"].mean("time"),
    slopes["icon"].sel(iwp=slice(1e-1, 10)),
    cf_change["icon"],
    feedbacks["icon"],
    xr.full_like(slopes["icon"], 0),
    feedback_cum_bs["icon"],
    err_feedback_bs["icon"],
    dim="iwp",
)
# fig.savefig("plots/diurnal_cycle/icon_2d_trend.png", dpi=300)

# %%check change in mean cf
for name in names:
    mean_cf_change = (
        (slopes[name] / 100 * cf[name].mean("time")).sum(dim=dim[name]).sum()
    )
    print(f"{name} mean cf change: {mean_cf_change.values} 1/K")

# %% plots for talk
fig, axes = plot_2d_trend_talk(
    cf["ccic"].mean("time"),
    slopes["ccic"],
    cf_change["ccic"],
    feedbacks["ccic"],
    p_values["ccic"],
    feedback_cum_bs["ccic"],
    err_feedback_bs["ccic"],
    dim="iwp",
)
fig.savefig(
    "plots/diurnal_cycle/talk/ccic_2d_trend_talk.png", bbox_inches="tight", dpi=300
)

# %%
fig, axes = plot_2d_trend_talk(
    cf["gpm"].mean("time"),
    slopes["gpm"],
    cf_change["gpm"],
    feedbacks["gpm"],
    p_values["gpm"],
    feedback_cum_bs["gpm"],
    err_feedback_bs["gpm"],
    dim="bt",
)
fig.savefig(
    "plots/diurnal_cycle/talk/gpm_2d_trend_talk.png", bbox_inches="tight", dpi=300
)

# %% compare cf era5 and CCIC
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].pcolormesh(
    cf["ccic"].mean("time").local_time,
    cf["ccic"].mean("time").iwp,
    cf["ccic"].mean("time").T,
    vmin=0,
    vmax=0.001,
)

axes[1].pcolormesh(
    cf["era5"].mean("time").local_time,
    cf["era5"].mean("time").iwp,
    cf["era5"].mean("time").T,
    vmin=0,
    vmax=0.002,
)

for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel("Local Time")
    ax.set_ylabel("IWP (kg/m^2)")
    ax.invert_yaxis()
    ax.set_ylim([10, 1e-1])

# %%
fig, ax = plt.subplots()

ax.plot(
    cf['ccic'].iwp,
    cf['ccic'].sel(time='2016').mean(['time']).sum('local_time'),
    color='k'
)
ax.plot(
    cf['era5'].iwp,
    cf['era5'].sel(time='2016').mean(['time']).sum('local_time'),
    color='r'
)

ax.set_xscale('log')
ax.set_ylim([0, 0.02])
# %%
