# %%
import xarray as xr 
import matplotlib.pyplot as plt

# %% open datasets 
path = "/work/bm1183/m301049/era5/monthly/"
convergence = xr.open_dataarray(f"{path}/convergence.nc", chunks={}, decode_timedelta=False)
pressure = xr.open_dataarray(f"{path}/p.nc", chunks={}, decode_timedelta=False)
temp = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc", chunks={})
# %% 
mean_conv = convergence.isel(time=0).mean(['latitude', 'longitude'])
mean_pressure = pressure.isel(time=0).mean(['latitude', 'longitude'])
# %%
fig, ax = plt.subplots()
ax.plot(mean_conv, mean_pressure)
ax.invert_yaxis()
ax.set_xlim(-0.5, 0.5)



# %%
