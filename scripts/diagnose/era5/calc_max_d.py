# %%
import dask
from dask.diagnostics import ProgressBar
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Add repository root to Python path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.helper_functions import read_era5_vars

dask.config.set(scheduler="synchronous")

# %%
ds = read_era5_vars()

# %% find maximum detrainment height
fig, ax = plt.subplots()
conv = (
    ds["convergence"]
    .isel(time=slice(210, 230), latitude=10, longitude=100)
    .sel(hybrid=slice(60, 95))
    .load()
)
p = (
    ds["pressure"]
    .isel(time=slice(210, 230), latitude=10, longitude=100)
    .sel(hybrid=slice(60, 95))
    .load()
)
for t in range(conv.time.size):
    ax.plot(conv.isel(time=t), p.isel(time=t) / 100, alpha=0.8)
ax.invert_yaxis()

# %% fit spline to conv before selecting the max
from scipy.interpolate import UnivariateSpline

fig, ax = plt.subplots()
conv = (
    ds["convergence"]
    .isel(time=slice(100, 110), latitude=10, longitude=100)
    .sel(hybrid=slice(None, 95))
    .load()
)
p = (
    ds["pressure"]
    .isel(time=slice(100, 110), latitude=10, longitude=100)
    .sel(hybrid=slice(None, 95))
    .load()
)
for t in range(conv.time.size):
    x = conv.isel(time=t).values
    y = p.isel(time=t).values / 100
    # Fit spline
    spline = UnivariateSpline(y, x, s=0)
    y_spline = np.linspace(y.min(), y.max(), 100)
    x_spline = spline(y_spline)
    ax.plot(x_spline, y_spline)
ax.invert_yaxis()


# %%
with ProgressBar():
    max_d_level = (
        ds.convergence.sel(hybrid=slice(60, 95)).idxmax(dim="hybrid").load()
    )  # disregard maxima at the edges of the domain
max_d_level.to_netcdf(
    "/work/bm1183/m301049/era5/monthly/level_of_max_convergence_60_95hPa.nc"
)

# %%
with ProgressBar():
    # Only select where max_d_level is not NaN
    max_d = (
        ds.convergence.sel(hybrid=max_d_level)
        .where((max_d_level > 60) & (max_d_level < 95))
        .load()
    )
max_d.to_netcdf("/work/bm1183/m301049/era5/monthly/max_convergence_60_95hPa.nc")

# %%
with ProgressBar():
    stability_d = (
        ds.stability.sel(hybrid=max_d_level)
        .where((max_d_level > 60) & (max_d_level < 95))
        .load()
    )
stability_d.to_netcdf(
    "/work/bm1183/m301049/era5/monthly/stability_at_max_convergence_60_95hPa.nc"
)
# %%
