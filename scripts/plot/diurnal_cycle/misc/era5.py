# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from src.helper_functions import load_histograms, interpolate_bins
from src.plot import definitions
import pickle

# %%
hist_era5 = {}
for region in ["sea", "all"]:
    hist_era5[region] = xr.open_dataset(
        f"/work/bm1183/m301049/era5/diagnosed/iwp_hist_monthly_interpolated_{region}.nc"
    )
hist_era5["land"] = hist_era5["all"] - hist_era5["sea"]
hists = load_histograms()
colors, line_labels, linestyles = definitions()
bins_lt = np.arange(0, 25, 1)
bins_iwp = np.logspace(-3, 2, 254)[::4]

# %% load icon histogram
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)

with open(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/jed0011_iwp_hist.pkl",
    "rb",
) as f:
    hist_icon = pickle.load(f)
    hist_icon = xr.DataArray(
        hist_icon,
        coords={"iwp": cre.iwp},
        dims=["iwp"],
    )
#  interpolate
hist_icon_int = interpolate_bins(hist_icon, bins_iwp, "iwp")

# %% load rcemip data
ds = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
)
ds["fwp"] = ds["fwp"] * 1e-3
rcemip_pdf = interpolate_bins(ds["f"].mean("model"), bins_iwp, "fwp")

# %% load xshield data
xshield_cont = xr.open_dataset(
    "/work/bm1183/m301049/xshield/xshield24v2_iw_histogram.nc"
)

# %% plot hist
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    hist_era5["all"].iwp,
    hist_era5["all"]["hist"].sel(time="2016").sum(["time", "local_time"])
    / hist_era5["all"]["size"].sel(time="2016").sum("time"),
    label="ERA5",
    color="#F60019",
    linestyle="--",
    linewidth=2,
)

ax.plot(
    hist_icon_int.iwp,
    hist_icon_int,
    label=line_labels["icon"],
    color=colors["icon"],
    linestyle=linestyles["icon"],
)
ax.plot(
    rcemip_pdf.iwp,
    rcemip_pdf.sel(SST=295),
    label=line_labels["rcemip"],
    color=colors["rcemip"],
    linestyle=linestyles["rcemip"],
)

ax.plot(
    xshield_cont.iwp,
    xshield_cont["f"],
    label=line_labels["xshield"],
    color=colors["xshield"],
    linestyle=linestyles["xshield"],
)

for key in ["dardar", "two_c_ice", "spare_ice", "ccic"]:
    ax.plot(
        hists[key].iwp,
        hists[key]["hist"].sel(time="2016").sum("time")
        / hists[key]["size"].sel(time="2016").sum("time"),
        label=line_labels[key],
        color=colors[key],
        linestyle=linestyles[key],
    )

ax.set_xscale("log")
ax.set_xlim([1e-3, 2e1])
ax.set_ylim(0, 0.015)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(frameon=False)
ax.set_ylabel(r"$P(I)$")
ax.set_xlabel(r"$I$ / kg m$^{-2}$")
ax.set_yticks([0, 0.006, 0.012])

fig.savefig("plots/anvil_thinning/talk/era5_dists.pdf", bbox_inches="tight")

# %% get same fraction of deep convective clouds from era5
fraction_dc_ccic = (
    hists["ccic"]["hist"].sel(time="2016", iwp=slice(1, None)).sum()
    / hists["ccic"]["size"].sel(time="2016").sum()
)

mean_hist = hist_era5["all"]['hist'].sel(time='2016').sum(['time', 'local_time']) / hist_era5["all"]['size'].sel(time='2016').sum('time')
# cumulative sum of hist starting from highest iwp
area_era5 = mean_hist.sortby("iwp", ascending=False).cumsum("iwp")
# find iwp where cumulative sum is equal to fraction of deep convective clouds in ccic
iwp_threshold = area_era5.where(area_era5 >= fraction_dc_ccic).dropna("iwp")["iwp"].max().item()
print(f"IWP threshold for deep convective clouds in ERA5: {iwp_threshold:.2f} kg m^-2")

# %% diurnal cycle era5
fig, ax = plt.subplots(figsize=(8, 5))
colors = {"sea": "b", "land": "brown"}
for region in ["sea", "land"]:
    ax.plot(
        hist_era5[region].local_time,
        hist_era5[region]["hist"]
        .sel(time="2016", iwp=slice(iwp_threshold, None))
        .sum(["time", "iwp"])
        / hist_era5[region]["size"].sel(time="2016").sum("time"),
        label=f"ERA5 {region}",
        color=colors[region],
    )

# %% 
# %% load albedo
albedo_iwp = xr.open_dataset("/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc")
albedo_bt = xr.open_dataset("/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc")
SW_in = xr.open_dataarray(
    "/work/bm1183/m301049/icon_hcap_data/publication/incoming_sw/SW_in_daily_cycle.nc"
)
SW_in = SW_in.interp(time_points=hists["ccic"]["local_time"], method="linear")
