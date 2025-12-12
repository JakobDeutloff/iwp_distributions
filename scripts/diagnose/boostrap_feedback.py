# %%
import xarray as xr
from src.helper_functions import (
    normalise_histograms,
    deseason,
    detrend_hist_2d,
    regress_hist_temp_2d,
)
from src.plot import definitions
from scipy.signal import detrend
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt


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
cutoffs = {
    "ccic": {"iwp": slice(1e-1, None)},
    "gpm": {"bt": slice(None, 260)},
    "icon": {"iwp": slice(1e-1, None)},
}
SW_in = xr.open_dataarray(
    "/work/bm1183/m301049/icon_hcap_data/publication/incoming_sw/SW_in_daily_cycle.nc"
)
SW_in = SW_in.interp(time_points=hists["ccic"]["local_time"], method="linear")
albedo = xr.open_dataset('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo.nc')['hc_albedo']

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc").t2m

# %%
hists_monthly = {}
for name in names:
    hists_monthly[name] = normalise_histograms(hists[name])

# %%  detrend and deseasonalize
hists_detrend = {}
temp_detrend = xr.DataArray(detrend(temp), coords=temp.coords, dims=temp.dims)
temp_detrend = deseason(temp_detrend)
for name in names:
    hists_detrend[name] = detrend_hist_2d(hists_monthly[name])
    hists_detrend[name] = deseason(hists_detrend[name])

# %%
def calc_feedback_bs(seed, name='ccic', len_block=70):

    n_sample = hists_detrend[name].time.size
    n_blocks = int(hists_monthly[name].time.size / len_block)
    max_idx_block = n_sample-len_block
    np.random.seed(seed)
    block_idxs = np.random.randint(0, max_idx_block, n_blocks)
    time_idx = []

    for i in block_idxs:
        time_idx.extend(
            list(
                range(i, i+len_block)
            )  # create list of time indices
        )
    slope_bs, _ = regress_hist_temp_2d(
        hists_detrend[name].isel(time=time_idx),
        temp_detrend,
        hists_monthly[name].isel(time=time_idx),
    )
    area_fraction_bs = hists[name]["hist"].isel(time=time_idx).sum("time") / hists[
        name
    ].isel(time=time_idx)["size"].sum(
        "time"
    )  # 1/1
    area_change_bs = (slope_bs / 100) * area_fraction_bs  # 1/K
    feedback_2d_bs = -1 * (
        (area_change_bs * SW_in * albedo) - ((area_change_bs) * SW_in * 0.1)
    )  # W / m^2 / K
    return feedback_2d_bs

# %% calc feedback ccic
n_iterations = 2000
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(
        tqdm(executor.map(calc_feedback_bs, range(n_iterations), ['ccic'] * n_iterations), total=n_iterations)
    )
#  put the results into an xarray
feedbacks = xr.concat(results, dim="iteration")
feedbacks.to_netcdf("/work/bm1183/m301049/diurnal_cycle_dists/ccic_bootstrap_feedback_2d.nc")

# %% calc feedback gpm
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(
        tqdm(executor.map(calc_feedback_bs, range(n_iterations), ['gpm'] * n_iterations), total=n_iterations)
    )
#  put the results into an xarray
feedbacks_gpm = xr.concat(results, dim="iteration")
feedbacks_gpm.to_netcdf("/work/bm1183/m301049/diurnal_cycle_dists/gpm_bootstrap_feedback_2d.nc")

# %% test bootstrapping 
n_iterations = [10, 30, 70, 100, 200, 300, 500, 750, 1000, 1500, 2000]
feedbacks_bs = {}
for n in n_iterations:
    print(f"Calculating bootstrap feedbacks for {n} iterations")
    with ProcessPoolExecutor(max_workers=128) as executor:
        results = list(
            tqdm(executor.map(calc_feedback_bs, range(n), ['ccic'] * n), total=n)
        )
    #  put the results into an xarray
    feedbacks = xr.concat(results, dim="iteration")
    feedbacks_bs[n] = feedbacks

mean_feedbacks = []
std_feedbacks = []
for n in n_iterations:
    mean_feedbacks.append(
        feedbacks_bs[n].sel(cutoffs['ccic']).sum(['local_time', 'iwp']).mean("iteration").values
    )
    std_feedbacks.append(
        feedbacks_bs[n].sel(cutoffs['ccic']).sum(['local_time', 'iwp']).std("iteration").values
    )

# %%
fig, axes = plt.subplots(2, 1, figsize=(6,4), sharex=True)
axes[0].plot(n_iterations, mean_feedbacks, 'o-', color='k')
axes[1].plot(n_iterations, std_feedbacks, 'o-', color='k')
axes[1].set_xlabel("Number of bootstrap iterations")
axes[0].set_ylabel("Mean Feedback / W m$^{-2}$ K$^{-1}$")
axes[1].set_ylabel("Std Feedback / W m$^{-2}$ K$^{-1}$")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
