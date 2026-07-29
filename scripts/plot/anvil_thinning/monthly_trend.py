# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, interpolate_bins, load_histograms
import pickle


# %% initialize containers
bins = np.logspace(-3, 2, 254)[::4]
colors, line_labels, linestyles = definitions()
hists = load_histograms()

# %% find right size threshold for 2c ice and dardar
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
bins_size = np.arange(0, np.max([hists["two_c_ice"]["size"].max(), hists['dardar']['size'].max()]), 1e5)
axes[0].hist(hists["two_c_ice"]["size"], color='k', bins=bins_size)
axes[1].hist(hists["dardar"]["size"], color='k', bins=bins_size)
axes[0].axvline(1.9e6, color='r', linestyle='--')
axes[1].axvline(1.9e6, color='r', linestyle='--')

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Number of Months")
axes[1].set_xlabel("Sample Size")
axes[0].set_title("2C-ICE")
axes[1].set_title("DARDAR")
# add letters 
for i, ax in enumerate(axes):
    ax.text(0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.savefig("plots/anvil_thinning/sample_size_histograms.pdf", bbox_inches="tight")

# %% number of months below sample size threshold 
for key in ['two_c_ice', 'dardar']:
    num_months = (hists[key]["size"] < 1.9e6).sum().item()
    print(f"{key}: {num_months} months below threshold, {num_months/len(hists[key]['size'])*100:.2f}% of months")

# %% filter 2c_ice and dadar data for size
hists['two_c_ice'] = hists['two_c_ice'].where(hists['two_c_ice']["size"] > 1.9e6)
hists['dardar'] = hists['dardar'].where(hists['dardar']["size"] > 1.9e6)

#%% normalise hists
hists_normalized = {}
for key in hists.keys():
    hists_normalized[key] = hists[key]["hist"] / hists[key]["size"]

# %% load era5 surface temp
t_mean = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %% plot all histograms to check how they look
plot_hists(
    hists_normalized["dardar"].sel(time=slice("2007-05", "2025-07")),
    t_mean.sel(time=slice("2007-05", "2025-07")),
    bins,
)

# %% detrend and deseasonalize monthly values
hists_deseason = {}

# temperature
t_detrend = xr.DataArray(detrend(t_mean), coords=t_mean.coords, dims=t_mean.dims)
t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)

for key in hists_normalized.keys():
    hists_detrend = nan_detrend(hists_normalized[key])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[key] = hists_deseason_ds

# %%regression
slopes_monthly = {}
error_montly = {}
p_vals_monthly = {}
for key in hists_deseason.keys():
    slopes_ds = []
    err_ds = []
    p_vals_ds = []
    hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
    temp = t_deseason.sel(time=hist_vals.time)
    for i in range(hists_deseason[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(temp.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
        p_vals_ds.append(res.pvalue)
    slopes_monthly[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    error_montly[key] = xr.DataArray(
        err_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )
    p_vals_monthly[key] = xr.DataArray(
        p_vals_ds,
        coords={"iwp": hists_deseason[key].iwp},
        dims=["iwp"],
    )



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

#  interpolate
iwp_hists_int = {}
for run in ["jed0011", "jed0022", "jed0033"]:
    iwp_hists_int[run] = interpolate_bins(iwp_hists[run], bins, "iwp")

cre["iwp"] = np.log10(cre["iwp"])
cre = cre.interp(
    iwp=np.log10(hists["ccic"].iwp), method="linear"
).drop_vars("iwp")
cre["iwp"] = hists["ccic"].iwp


iwp_change_icon = {}
temp_deltas = {"jed0022": 4, "jed0033": 2}
for run in ["jed0022", "jed0033"]:
    iwp_change_icon[run] = (
        iwp_hists_int[run] - iwp_hists_int["jed0011"]
    ) / temp_deltas[run]
iwp_change_icon_mean = (iwp_change_icon["jed0022"] + iwp_change_icon["jed0033"]) / 2
slopes_monthly['icon'] = iwp_change_icon_mean
slopes_monthly['icon_2K'] = iwp_change_icon["jed0033"]
slopes_monthly['icon_4K'] = iwp_change_icon["jed0022"]
hists_normalized['icon'] = iwp_hists_int["jed0011"]
hists_normalized['icon_2K'] = iwp_hists_int["jed0011"]
hists_normalized['icon_4K'] = iwp_hists_int["jed0011"]

# %% load rcemip data
ds = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
)
ds["fwp"] = ds["fwp"] * 1e-3
rcemip_pdf = interpolate_bins(ds["f"].mean("model"), bins, "fwp")
diff_rcemip = (rcemip_pdf.sel(SST=305) - rcemip_pdf.sel(SST=295)) / 10
slopes_monthly['rcemip'] = diff_rcemip
hists_normalized['rcemip'] = rcemip_pdf.sel(SST=295)

# %% load xshield data 
xshield_cont = xr.open_dataset("/work/bm1183/m301049/xshield/xshield24v2_iw_histogram.nc")
xshield_4k = xr.open_dataset("/work/bm1183/m301049/xshield/xshield24v2_PLUS_4K_iw_histogram.nc")
diff_xshield = (xshield_4k - xshield_cont) / 4
slopes_monthly['xshield'] = diff_xshield['f']
hists_normalized['xshield'] = xshield_cont['f']


# %% calculate feedback
feedback = {}
for key in slopes_monthly.keys():
    feedback[key] = slopes_monthly[key] * cre["net"].values

# %% partition feedback into area and opacity feedback
feedback_area = {}
feedback_opacity = {}
for key in hists.keys():
    g_cap = (slopes_monthly[key]).sum() / (hists_normalized[key].mean('time')).sum()
    print(f"g_cap for {key}: {g_cap*100} %/K")
    g_prime = (
        (slopes_monthly[key]) / hists_normalized[key].mean('time')
    ) - g_cap
    feedback_area[key] = (cre['net']*hists_normalized[key].mean('time')).sum() * g_cap 
    feedback_opacity[key] = (
        g_prime * hists_normalized[key].mean('time') * cre["net"]
    ).sum()

for key in ['icon', 'rcemip', 'icon_2K', 'icon_4K', 'xshield']:
    g_cap = (slopes_monthly[key]).sum() / (hists_normalized[key]).sum()
    print(f"g_cap for {key}: {g_cap*100} %/K")
    g_prime = (
        (slopes_monthly[key]) / hists_normalized[key]
    ) - g_cap
    feedback_area[key] = (cre['net']*hists_normalized[key].values).sum() * g_cap.values
    feedback_opacity[key] = (
        g_prime.values * hists_normalized[key].values * cre["net"]
    ).sum()

# %% plot all distributions and cre for 2016 
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1])

axes[0].plot(
    iwp_hists_int['jed0011'].iwp,
    iwp_hists_int["jed0011"],
    label=line_labels['icon'],
    color=colors['icon'],
    linestyle=linestyles['icon'],
)
axes[0].plot(
    rcemip_pdf.iwp,
    rcemip_pdf.sel(SST=295),
    label=line_labels['rcemip'],
    color=colors['rcemip'],
    linestyle=linestyles['rcemip'],
)

axes[0].plot(
    xshield_cont.iwp,
    xshield_cont['f'],
    label=line_labels['xshield'],
    color=colors['xshield'],
    linestyle=linestyles['xshield'],
)

for key in ["dardar", "two_c_ice", "spare_ice", "ccic"]:
    axes[0].plot(
        hists[key].iwp,
        hists[key]['hist'].sel(time="2016").sum('time') / hists[key]['size'].sel(time="2016").sum('time'),
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

axes[0].set_xscale("log")
axes[0].set_xlim([1e-3, 2e1])
axes[0].set_ylim(0, 0.013)

axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].plot(
    cre.iwp,
    cre['net'],
    color='k',
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False)
axes[0].set_ylabel(r"$P(I)$")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[1].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[1].set_yticks([-100, 0, 40])
axes[0].set_yticks([0, 0.006, 0.012])

#add letters
for i, ax in enumerate(axes):
    ax.text(0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.savefig("plots/anvil_thinning/talk/distributions_cre_2016_icon_rce_x_obs_mach.pdf", bbox_inches="tight")

# %% plot slopes and p-value
fig, axes = plt.subplots(2, 1,figsize=(8, 6), sharex=True, height_ratios=[3, 1])

axes[0].plot(
    iwp_change_icon_mean.iwp,
    iwp_change_icon_mean,
    label=line_labels['icon'],
    color=colors['icon'],
    linestyle="--",
)

axes[0].plot(
    diff_rcemip.iwp,
    diff_rcemip,
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)

axes[0].plot(
    diff_xshield.iwp,
    diff_xshield.f,
    label=line_labels["xshield"],
    color=colors["xshield"],
    linestyle=linestyles["xshield"],
)

for key in hists.keys():
    axes[0].plot(
        slopes_monthly[key].iwp,
        slopes_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )
    axes[1].plot(
        p_vals_monthly[key].iwp,
        p_vals_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )

axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xscale("log")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)  
    ax.set_xlim(1e-3, 2e1)

axes[0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[1].set_ylabel("p-value")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[0].set_ylim(-0.00061, 0.0002)
axes[1].set_yticks([0.05, 0.5, 1])
axes[1].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3, bbox_to_anchor=(0.75, 0))

# add letters
for i, ax in enumerate(axes):
    ax.text(0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.savefig("plots/anvil_thinning//slopes_monthly.pdf", bbox_inches="tight")

# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])
markers = {
    "icon": "x",
    "rcemip": "x",
    "xshield": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
}

members = ['icon', 'rcemip', 'xshield', 'dardar', 'two_c_ice', 'spare_ice', 'ccic']
for key in members:
    axes[0].plot(
        feedback[key].iwp,
        feedback[key],
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

    axes[1].scatter(
        0,
        feedback[key].sum().item()/2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
    )
    axes[1].scatter(
        1,
        feedback_area[key].item()/2,
        color=colors[key],
        marker=markers[key],
    )
    axes[1].scatter(
        2,
        feedback_opacity[key].item()/2,
        color=colors[key],
        marker=markers[key],
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False, loc="upper left")
axes[0].set_yticks([-0.02, 0, 0.02])
axes[0].set_ylim(-0.022, 0.03)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xlim(-0.5, 2.5)
axes[1].set_xticklabels(["Total", "Area", "Opacity"], rotation=45)
axes[1].set_ylabel(r"$\lambda_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.02, 0, 0.05, 0.1])
axes[1].set_ylim(-0.035, 0.1)


handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(1.1, 0.98))

# # add letters
# for i, ax in enumerate(axes):
#     ax.text(0.03, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig("plots/anvil_thinning/talk/feedback_monthly_icon_rce_x_obs.pdf", bbox_inches="tight")

# %% calculate mean and std of feedback 
mean_feedback = np.mean([feedback[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
std_feedback = np.std([feedback[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
print(f"Mean feedback: {mean_feedback:.4f} W m^-2 K^-1")
print(f"Std feedback: {std_feedback:.4f} W m^-2 K^-1")
print(f"Feedback for ICON: {feedback['icon'].sum().item()/2:.4f} W m^-2 K^-1")
print(f"Feedback for RCEMIP: {feedback['rcemip'].sum().item()/2:.4f} W m^-2 K^-1")

# %% caclculate feedback fro every satellite
total_feedback = [feedback[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']]

# %% plot for thesis 
offsets = {
    "icon_2K": 0.2,
    "icon_4K": 0.3,
    "rcemip": 0.4,
    "ccic": 0.5,
    "two_c_ice": 0.6,
    "dardar": 0.7,
    "spare_ice": 0.8,
}
markers = {
    "icon_2K": "x",
    "icon_4K": "x",
    "rcemip": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
}
colors['icon_2K'] = '#1f948a'
colors['icon_4K'] = '#c1df24'
line_labels['icon_2K'] = "ICON +2K"
line_labels['icon_4K'] = "ICON +4K" 
linestyles['icon_2K'] = "--"
linestyles['icon_4K'] = "--"

fig, axes = plt.subplots(3, 2, figsize=(10, 8), height_ratios=[1, 0.3, 1], width_ratios=[1, 0.1], sharex='col')
axes[2, 0].axhline(0, color="k", linewidth=0.7)
axes[1, 0].axhline(0.05, color="k", linewidth=0.7)
# plot regression 
axes[0, 0].plot(
    iwp_change_icon['jed0033'].iwp,
    iwp_change_icon['jed0033'],
    label=line_labels['icon_2K'],
    color=colors['icon_2K'],
    linestyle="--",
)
axes[0, 0].plot(
    iwp_change_icon['jed0022'].iwp,
    iwp_change_icon['jed0022'],
    label=line_labels['icon_4K'],
    color=colors['icon_4K'],
    linestyle="--",
)


axes[0, 0].plot(
    diff_rcemip.iwp,
    diff_rcemip,
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)

for key in hists.keys():
    axes[0, 0].plot(
        slopes_monthly[key].iwp,
        slopes_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )
    axes[1, 0].plot(
        p_vals_monthly[key].iwp,
        p_vals_monthly[key],
        label=line_labels[key],
        color=colors[key],
    )

axes[0, 0].axhline(0, color="k", linewidth=0.5)

# plot feedback
members = offsets.keys()
for key in members:
    axes[2, 0].plot(
        feedback[key].iwp,
        feedback[key]/2,
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

    axes[2, 1].scatter(
        0,
        feedback[key].sum().item()/2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
    )

axes[0, 0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[0, 0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[1, 0].set_ylabel("p-value")
axes[1, 0].set_yticks([0.05, 0.5, 1])


axes[2, 0].set_ylabel(r"$\lambda_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[2, 0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[2, 0].set_yticks([-0.01, 0, 0.01])


axes[2, 1].spines[['bottom']].set_visible(False)
axes[2, 1].set_xticks([])
axes[2, 1].set_ylabel(r"$\lambda_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[2, 1].set_yticks([0, 0.05, 0.1, 0.15])
axes[2, 1].set_ylim([-0.035, 0.17])


for ax in axes[:, 0]: 
    ax.set_xlim(1e-3, 2e1)
    ax.set_xscale("log")
for ax in axes.flatten():
    ax.spines[["top", "right"]].set_visible(False)

# add letters
axes[0, 1].remove()
axes[1, 1].remove()
for i, ax in enumerate([axes[0, 0], axes[1, 0], axes[2, 0], axes[2, 1]]):
    ax.text(0.02, 0.95, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout() 

# add legends 
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(0.99, 0.9))
handles, labels = axes[2, 1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(0.99, 0.7))
fig.savefig("plots/thesis/feedback_monthly_thesis.pdf", bbox_inches="tight")
# %% print feedback values for table
for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar', 'icon_2K', 'icon_4K', 'rcemip']:
    print(f"{key}: {feedback[key].sum().item()/2:.3f} W m^-2 K^-1")

# %% plots for talk 
#  plot all distributions
fig, ax = plt.subplots(figsize=(8, 4))

colors['icon_control']  = "#462d7b"
colors['icon_2K'] = '#1f948a'
colors['icon_4K'] = '#c1df24'
line_labels['icon_2K'] = "ICON +2K"
line_labels['icon_4K'] = "ICON +4K" 
line_labels['icon_control'] = "ICON"
linestyles['icon_2K'] = "--"
linestyles['icon_4K'] = "--"
linestyles['icon_control'] = "--"

ax.plot(
    iwp_hists_int['jed0011'].iwp,
    iwp_hists_int["jed0011"],
    label=line_labels['icon_control'],
    color=colors['icon_control'],
    linestyle=linestyles['icon_control'],
)
ax.plot(
    rcemip_pdf.iwp,
    rcemip_pdf.sel(SST=295),
    label=line_labels['rcemip'],
    color=colors['rcemip'],
    linestyle=linestyles['rcemip'],
)


for key in ["dardar", "two_c_ice", "spare_ice", "ccic"]:
    ax.plot(
        hists[key].iwp,
        hists[key]['hist'].sel(time="2016").sum('time') / hists[key]['size'].sel(time="2016").sum('time'),
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )



ax.set_xscale("log")
ax.set_xlim([1e-3, 2e1])
ax.set_ylim(0, 0.013)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(frameon=False)
ax.set_ylabel(r"$P(I)$")
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
ax.set_yticks([0, 0.006, 0.012])

fig.savefig("plots/anvil_thinning/talk/distributions_icon_rce_obs.pdf", bbox_inches="tight")

# %%
fig, axes = plot_regression(
    temp=t_deseason,
    hists=hists_deseason["ccic"],
    slopes=slopes_monthly["ccic"],
    error=error_montly["ccic"],
    title='CCIC'
)
fig.savefig("plots/anvil_thinning/talk/regression_ccic.pdf", bbox_inches="tight")

# %% plot slopes and p-value
fig, axes = plt.subplots(2, 1,figsize=(8, 6), sharex=True, height_ratios=[3, 1])

axes[0].plot(
    iwp_change_icon['jed0033'].iwp,
    iwp_change_icon['jed0033'],
    label=line_labels['icon_2K'],
    color=colors['icon_2K'],
    linestyle="--",
)

axes[0].plot(
    iwp_change_icon['jed0022'].iwp,
    iwp_change_icon['jed0022'],
    label=line_labels['icon_4K'],
    color=colors['icon_4K'],
    linestyle="--",
)

axes[0].plot(
    diff_rcemip.iwp,
    diff_rcemip,
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle="--",
)


# for key in hists.keys():
#     axes[0].plot(
#         slopes_monthly[key].iwp,
#         slopes_monthly[key],
#         label=line_labels[key],
#         color=colors[key],
#     )
#     axes[1].plot(
#         p_vals_monthly[key].iwp,
#         p_vals_monthly[key],
#         label=line_labels[key],
#         color=colors[key],
#     )

axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xscale("log")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)  
    ax.set_xlim(1e-3, 2e1)

axes[0].set_ylabel(r"d$P(I)$/d$T$ / K$^{-1}$")
axes[1].set_ylabel("p-value")
axes[1].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].set_yticks([-0.0006, -0.0002, 0, 0.0002])
axes[0].set_ylim(-0.00061, 0.0002)
axes[1].set_yticks([0.05, 0.5, 1])
axes[1].axhline(0.05, color="k", linewidth=0.5)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, frameon=False, loc='lower right')

fig.savefig("plots/anvil_thinning/talk/slopes_monthly_icon_rce.pdf", bbox_inches="tight")
# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.4])
offsets = {
    "icon_2K": 0.2,
    "icon_4K": 0.3,
    "rcemip": 0.4,
    "ccic": 0.5,
    "two_c_ice": 0.6,
    "dardar": 0.7,
    "spare_ice": 0.8,
}
markers = {
    "icon_2K": "x",
    "icon_4K": "x",
    "rcemip": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
}
colors['icon_2K'] = '#1f948a'
colors['icon_4K'] = '#c1df24'
line_labels['icon_2K'] = "ICON +2K"
line_labels['icon_4K'] = "ICON +4K" 
linestyles['icon_2K'] = "--"
linestyles['icon_4K'] = "--"

members = ['icon_2K', 'icon_4K', 'rcemip', 'ccic', 'two_c_ice', 'dardar', 'spare_ice']
for key in members:
    axes[0].plot(
        feedback[key].iwp,
        feedback[key],
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

    axes[1].scatter(
        0,
        feedback[key].sum().item()/2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda_{\mathrm{P}}(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False, loc="upper left")
axes[0].set_yticks([-0.02, 0, 0.02])
axes[0].set_ylim(-0.022, 0.03)

axes[1].spines[['bottom']].set_visible(False)
axes[1].set_xticks([])
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylabel(r"$\lambda_{\mathrm{P}}$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.02, 0, 0.05, 0.1])
axes[1].set_ylim(-0.035, 0.1)

# add sherwood estimate
axes[1].errorbar(
    1, 
    -0.2,
    yerr=0.2,
    marker='d',
    color='#0077b6',
    capsize=5,
    label="Sherwood et al. (2020)"
)
axes[1].set_yticks([-0.4, -0.2, -0.02, 0, 0.05, 0.1])
axes[1].set_ylim(-0.4, 0.1)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(1.2, 0.98))

fig.tight_layout()
fig.savefig("plots/anvil_thinning/talk/feedback_monthly_all_sh.pdf", bbox_inches="tight")

# %% plot feedback with sherwood estimate 
fig, ax = plt.subplots(figsize=(1, 5))


members = ['icon_2K', 'icon_4K', 'rcemip', 'ccic', 'two_c_ice', 'dardar', 'spare_ice']
for key in members:
    ax.scatter(
        0,
        feedback[key].sum().item()/2,
        color=colors[key],
        marker=markers[key],
        label=line_labels[key],
    )

ax.errorbar(
    1, 
    -0.2,
    yerr=0.2,
    marker='d',
    color='#0077b6',
    capsize=5,
)
ax.set_xticks([0, 1])
ax.set_xlim(-0.5, 1.5)
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.set_yticks([-0.4, -0.2, -0.02, 0, 0.05, 0.1])

# %%
