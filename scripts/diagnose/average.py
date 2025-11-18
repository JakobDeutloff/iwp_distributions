# %%
import xarray as xr
import numpy as np

# %%
path = "/work/bm1183/m301049/era5/monthly"
t = xr.open_dataarray(
    f"{path}/t.nc",
).load()
#%% latitude weighted mean 
weights = np.cos(np.deg2rad(t.latitude))
t_weighted = t.weighted(weights)
t_mean = t_weighted.mean(dim=["latitude", "longitude"])
t_mean.to_netcdf(f"{path}/t_avg.nc")
# %%
p = xr.open_dataarray(
    f"{path}/pressure_latlon.nc",
).load()
p.mean(dim=["latitude", "longitude"]).to_netcdf(f"{path}/p_avg.nc")
p.delete()

# %%
sw = xr.open_dataarray(
    f"{path}/235003_fc_latlon.grb",
    engine="cfgrib",
).load()
sw.mean(dim=["latitude", "longitude"]).to_netcdf(f"{path}/sw_avg.nc")
sw.delete()
# %%
lw = xr.open_dataarray(
    f"{path}/235004_fc_latlon.grb",
    engine="cfgrib",
)
lw.mean(dim=["latitude", "longitude"]).to_netcdf(f"{path}/lw_avg.nc")

# %%
