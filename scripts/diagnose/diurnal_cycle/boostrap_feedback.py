# %%
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
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
    "gpm": {"bt": slice(190, 260)},
}
SW_in = xr.open_dataarray(
    "/work/bm1183/m301049/icon_hcap_data/publication/incoming_sw/SW_in_daily_cycle.nc"
)
SW_in = SW_in.interp(time_points=hists["ccic"]["local_time"], method="linear")
albedo_iwp = xr.open_dataset('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc')['hc_albedo']
albedo_bt = xr.open_dataset('/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc')['hc_albedo']
albedo = {
    "ccic": albedo_iwp,
    "gpm": albedo_bt,
}

# %% cut data to relevant ranges
for name in names:
    hists[name] = hists[name].sel(cutoffs[name])
    albedo[name] = albedo[name].sel(cutoffs[name])

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

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
def calc_feedback_bs(seed, name='ccic', len_block=36):

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
        (area_change_bs * SW_in * albedo[name]) - ((area_change_bs) * SW_in * 0.1)
    )  # W / m^2 / K
    return feedback_2d_bs

# %% calc feedback ccic
n_iterations = 2000
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(
        tqdm(executor.map(calc_feedback_bs, range(n_iterations), ['ccic'] * n_iterations), total=n_iterations)
    )
feedbacks = xr.concat(results, dim="iteration")
feedbacks.to_netcdf("/work/bm1183/m301049/diurnal_cycle_dists/ccic_bootstrap_feedback_2d.nc")

# %% calc feedback gpm
with ProcessPoolExecutor(max_workers=128) as executor:
    results = list(
        tqdm(executor.map(calc_feedback_bs, range(n_iterations), ['gpm'] * n_iterations), total=n_iterations)
    )
feedbacks_gpm = xr.concat(results, dim="iteration")
feedbacks_gpm.to_netcdf("/work/bm1183/m301049/diurnal_cycle_dists/gpm_bootstrap_feedback_2d.nc")

# %%
