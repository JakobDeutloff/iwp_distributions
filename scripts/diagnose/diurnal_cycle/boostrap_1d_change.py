# %%
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import xarray as xr
from src.helper_functions import (
    regress_hist_temp_1d,
    nan_detrend,
    deseason,
)
from scipy.signal import detrend
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# %% load ccic and gpm data
hists = {
    'ccic': {},
    'gpm': {}
}
regions = ['all', 'land', 'sea']
datasets = ['ccic', 'gpm']
for ds in datasets:
    for region in regions:
        hists[ds][region] = xr.open_dataset(
            f"/work/bm1183/m301049/diurnal_cycle_dists/{ds}_2d_monthly_{region}.nc"
        )

# %% calculate cloud fraction
cf = {
    'ccic': {},
    'gpm': {}
}

for ds in datasets:
    for region in regions:
        if ds == 'ccic':
            cf[ds][region] = hists[ds][region]["hist"].sel(iwp=slice(1, None)).sum(
                "iwp"
            ) / hists[ds][region]["hist"].sum(["iwp", "local_time"])
        else:
            cf[ds][region] = hists[ds][region]["hist"].sel(bt=slice(None, 231)).sum(
                "bt"
            ) / hists[ds][region]["hist"].sum(["bt", "local_time"])

# %% normalise  cloud fractions
cf_norm = {
    'ccic': {},
    'gpm': {}
}
for ds in datasets:
    for region in regions:
        cf_norm[ds][region] = cf[ds][region] / cf[ds][region].sum("local_time")


# %% load era5 surface temp
temps = {}
temps["all"] = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m
temps["sea"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc"
).t2m
temps["land"] = xr.open_dataset(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics_land.nc"
).t2m

# %% detrend and deseasonalize
temps_deseason = {}
for region in regions:
    temp_detrend = xr.DataArray(
        detrend(temps[region]), coords=temps[region].coords, dims=temps[region].dims
    )
    temps_deseason[region] = deseason(temp_detrend)
cf_deseason = {
    'ccic': {},
    'gpm': {}
}
for ds in datasets:
    for region in regions:
        cf_detrend = nan_detrend(cf_norm[ds][region], dim="local_time")
        cf_deseason[ds][region] = deseason(cf_detrend)

# %% regression

def calc_regression_bs(seed, dataset='ccic', region='all', len_block=36):

    n_sample = cf[dataset][region].time.size
    n_blocks = int(cf[dataset][region].time.size / len_block)
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
    
    slopes, _ = regress_hist_temp_1d(
        cf_deseason[dataset][region].isel(time=time_idx), temps_deseason[region], cf_norm[dataset][region]
    )
    return slopes

# %% calc slopes
n_iterations = 2000
for dataset in datasets:
    for region in regions:
        with ProcessPoolExecutor(max_workers=128) as executor:
            results = list(
                tqdm(executor.map(calc_regression_bs, range(n_iterations), [dataset] * n_iterations, [region] * n_iterations), total=n_iterations)
            )
        slopes = xr.concat(results, dim="iteration")
        slopes.to_netcdf(f"/work/bm1183/m301049/diurnal_cycle_publication/{dataset}_{region}_bootstrap_slopes_1d.nc")


# %%
