# %% import
import xarray as xr
from dask.diagnostics import ProgressBar

# %%
path = "/work/bm1183/m301049/era5/monthly"
chunks = {'longitude': 10}
t = xr.open_dataarray(
    f"{path}/t.nc",
    chunks=chunks,
)
p = xr.open_dataarray(
    f"{path}/pressure_latlon.nc",
    chunks=chunks,
)
sw = xr.open_dataarray(
    f"{path}/sw.nc",
    chunks=chunks,
)
lw = xr.open_dataarray(
    f"{path}/lw.nc",
    chunks=chunks,
)

# %% define parameters
R = 8.314  # J/mol/K
cp = 29.07  # J/mol/K

# calculate iris quantities - save and reload since xarray otherwise keeps chunks in memory which leads to memory error
seconds_of_day = 24 * 60 * 60

rad_tendency = -(sw + lw) * seconds_of_day  # K/day
with ProgressBar():
    rad_tendency.to_netcdf(f"{path}/net_rad_tendency.nc")
rad_tendency = xr.open_dataarray(
    f"{path}/net_rad_tendency.nc",
    chunks=chunks,
)

# %%
dt_dp = t.differentiate("hybrid") / p.differentiate("hybrid")  # K/hPa
with ProgressBar():
    dt_dp.to_netcdf(f"{path}/dt_dp.nc")
dt_dp = xr.open_dataarray(
    f"{path}/dt_dp.nc", 
    chunks=chunks,
)
stability = (t / p) * (R / cp) - dt_dp  # K/hPa
with ProgressBar():
    stability.to_netcdf(f"{path}/stability.nc")
stability = xr.open_dataarray(
    f"{path}/stability.nc", 
    chunks=chunks,
)

w_r = rad_tendency / stability  # hPa/day
with ProgressBar():
    w_r.to_netcdf(f"{path}/w_r.nc")
w_r = xr.open_dataarray(
    f"{path}/w_r.nc", 
    chunks=chunks,
)

conv = w_r.differentiate("hybrid") / p.differentiate("hybrid")  # 1/day
with ProgressBar():
    conv.to_netcdf(f"{path}/conv.nc")
conv = xr.open_dataarray(
    f"{path}/conv.nc",
    chunks=chunks,
)

# %% set attributes
rad_tendency.attrs = {
    "long_name": "Clear Sky Radiative tendency",
    "units": "K/day",
}
dt_dp.attrs = {
    "long_name": "Lapse Rate",
    "units": "K/hPa",
}
stability.attrs = {
    "long_name": "Static stability",
    "units": "K/hPa",
}
w_r.attrs = {
    "long_name": "Radiatively Driven Subsidence Velocity",
    "units": "hPa/day",
}
conv.attrs = {
    "long_name": "Convergence of Radiatively Driven Subsidence Velocity",
    "units": "1/day",
}

# %%
with ProgressBar():
    print("Saving rad tendency")
    rad_tendency.to_netcdf(f"{path}/rad_tendency.nc")
    print("Saving dt/dp")
    dt_dp.to_netcdf(f"{path}/lapse_rate.nc")
    print("Saving stability")
    stability.to_netcdf(f"{path}/stability.nc")
    print("Saving w_r")
    w_r.to_netcdf(f"{path}/subsidence.nc")
    print("Saving conv")
    conv.to_netcdf(f"{path}/convergence.nc")
