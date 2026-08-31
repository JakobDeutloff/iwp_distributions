# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.helper_functions import nan_detrend, load_histograms

# %%  load data
hists = load_histograms("obs")
hists_model = load_histograms("model")

# %% find right size threshold for 2c ice and dardar
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True)
bins_size = np.arange(
    0, np.max([hists["two_c_ice"]["size"].max(), hists["dardar"]["size"].max()]), 1e5
)
axes[0].hist(hists["two_c_ice"]["size"], color="k", bins=bins_size)
axes[1].hist(hists["dardar"]["size"], color="k", bins=bins_size)
axes[0].axvline(1.9e6, color="r", linestyle="--")
axes[1].axvline(1.9e6, color="r", linestyle="--")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Number of Months")
axes[1].set_xlabel("Sample Size")
axes[0].set_title("2C-ICE")
axes[1].set_title("DARDAR")
# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight="bold"
    )
fig.savefig(
    "plots/anvil_thinning/publication/sample_size_histograms.pdf", bbox_inches="tight"
)

# %% number of months below sample size threshold
for key in ["two_c_ice", "dardar"]:
    num_months = (hists[key]["size"] < 1.9e6).sum().item()
    print(
        f"{key}: {num_months} months below threshold, {num_months/len(hists[key]['size'])*100:.2f}% of months"
    )

# %% filter 2c_ice and dadar data for size
hists["two_c_ice"] = hists["two_c_ice"].where(hists["two_c_ice"]["size"] > 1.9e6)
hists["dardar"] = hists["dardar"].where(hists["dardar"]["size"] > 1.9e6)

# %% normalise hists
hists_normalized = {}
for key in hists.keys():
    hists_normalized[key] = hists[key]["hist"] / hists[key]["size"]

# %% load era5 surface temp
t_mean = xr.open_dataset("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").t2m

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
slopes = xr.Dataset()
error = xr.Dataset()
p_vals = xr.Dataset()
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
    slopes = slopes.assign(
        {
            key: xr.DataArray(
                slopes_ds, coords={"iwp": hists_deseason[key].iwp}, dims=["iwp"]
            )
        }
    )
    error = error.assign(
        {
            key: xr.DataArray(
                err_ds, coords={"iwp": hists_deseason[key].iwp}, dims=["iwp"]
            )
        }
    )
    p_vals = p_vals.assign(
        {
            key: xr.DataArray(
                p_vals_ds,
                coords={"iwp": hists_deseason[key].iwp},
                dims=["iwp"],
            )
        }
    )

# %% calc slopes of models
slopes = slopes.assign(
    {"rcemip": (hists_model["rcemip_plus10K"] - hists_model["rcemip_control"]) / 10}
)
slopes = slopes.assign(
    {
        "icon_ap": (
            ((hists_model["icon_ap_plus4K"] - hists_model["icon_ap_control"]) / 4)
            + ((hists_model["icon_ap_plus2K"] - hists_model["icon_ap_control"]) / 2)
        )
        / 2
    }
)
slopes = slopes.assign(
    {
        'icon_ap_plus2K': (hists_model["icon_ap_plus2K"] - hists_model["icon_ap_control"]) / 2
    }
)
slopes = slopes.assign(
    {
        'icon_ap_plus4K': (hists_model["icon_ap_plus4K"] - hists_model["icon_ap_control"]) / 4
    }
)
slopes = slopes.assign(
    {"xshield": (hists_model["xshield_plus4K"] - hists_model["xshield_control"]) / 4.25}
)
slopes = slopes.assign(
    {
        "icon_amip": (
            hists_model["icon_amip_plus4K"] - hists_model["icon_amip_control"]
        )
        / 4.25
    }
)

# %% set attributes of slopes and errors
slopes.attrs["units"] = "1/K"
error.attrs["units"] = "1/K"
p_vals.attrs["units"] = ""
slopes.attrs["description"] = (
    "Change in ice water path frequency per degree of tropical mean surface warming"
)
error.attrs["description"] = (
    "Standard error of the slopes of monthly deseasonalized and detrended histograms vs. deseasonalized and detrended surface temperature"
)
p_vals.attrs["description"] = (
    "P-values of slopes of monthly deseasonalized and detrended histograms vs. deseasonalized and detrended surface temperature"
)
# %% save slopes and errors
slopes.to_netcdf("/work/bu1562/m301049/iwp_dists/slopes_monthly.nc")
error.to_netcdf("/work/bu1562/m301049/iwp_dists/errors_monthly.nc")
p_vals.to_netcdf("/work/bu1562/m301049/iwp_dists/p_vals_monthly.nc")

# %%
