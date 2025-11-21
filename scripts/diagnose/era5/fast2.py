# %% Imports
import xarray as xr
from dask.diagnostics import ProgressBar

# %% Paths and chunking
path = "/work/bm1183/m301049/era5/monthly"

# Use spatial chunking so each chunk fits comfortably in memory (≈100–300 MB)
chunks = {"longitude": 10, "hybrid": -1, 'time': -1, 'latitude': -1}

# %% Load data lazily
t = xr.open_dataarray(f"{path}/t.nc", chunks=chunks)
p = xr.open_dataarray(f"{path}/pressure_latlon.nc", chunks=chunks)
sw = xr.open_dataarray(f"{path}/sw.nc", chunks=chunks)
lw = xr.open_dataarray(f"{path}/lw.nc", chunks=chunks)

# %% Constants
R = 8.314  # J/mol/K
cp = 29.07  # J/mol/K
seconds_of_day = 24 * 60 * 60

# %% Derived quantities
rad_tendency = (-(sw + lw) * seconds_of_day)
dp = p.differentiate("hybrid")
dt = t.differentiate("hybrid")
dt_dp = (dt / dp)
stability = ((t / p) * (R / cp) - dt_dp)
w_r = (rad_tendency / stability)
conv = (w_r.differentiate("hybrid") / dp)

# %% Metadata
rad_tendency.attrs = {"long_name": "Clear Sky Radiative tendency", "units": "K/day"}
rad_tendency.name = "net_rad_tendency"
dt_dp.attrs = {"long_name": "Lapse Rate", "units": "K/hPa"}
dt_dp.name = "lapse_rate"
stability.attrs = {"long_name": "Static stability", "units": "K/hPa"}
stability.name = "stability"
w_r.attrs = {"long_name": "Radiatively Driven Subsidence Velocity", "units": "hPa/day"}
w_r.name = "subsidence"
conv.attrs = {"long_name": "Convergence of Radiatively Driven Subsidence Velocity", "units": "1/day"}
conv.name = "convergence"

# %% Save to Zarr — fast, parallelized on one machine
datasets = {
    "rad_tendency": rad_tendency,
    "lapse_rate": dt_dp,
    "stability": stability,
    "subsidence": w_r,
    "convergence": conv,
}

with ProgressBar():
    for name, arr in datasets.items():
        print(f"Saving {name}")
        encoding = {arr.name: {"chunksizes": [arr.sizes[dim] if dim != "longitude" else 10 for dim in arr.dims]}}
        arr.to_netcdf(f"{path}/{name}.nc", encoding=encoding)
# %%
