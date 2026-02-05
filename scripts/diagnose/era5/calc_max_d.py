# %%
import dask 
from dask.diagnostics import ProgressBar
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.helper_functions import read_era5_vars
dask.config.set(scheduler='synchronous')

# %% 
ds = read_era5_vars()

# %% find maxima of convergence between 50 and 90 hPa
with ProgressBar():
    max_d = ds.convergence.sel(hybrid=slice(50, 90)).max(dim='hybrid').load()
max_d.to_netcdf("/work/bm1183/m301049/era5/monthly/max_convergence_50_90hPa.nc")

# %%
with ProgressBar():
    max_d_level = ds.convergence.sel(hybrid=slice(50, 90)).idxmax(dim='hybrid').load()
max_d_level.to_netcdf("/work/bm1183/m301049/era5/monthly/level_of_max_convergence_50_90hPa.nc")
# %%
with ProgressBar():
    stability_d = ds.stability.sel(hybrid=max_d_level).load()
stability_d.to_netcdf("/work/bm1183/m301049/era5/monthly/stability_at_max_convergence_50_90hPa.nc")