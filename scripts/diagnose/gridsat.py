# %%
import xarray as xr 

# %% 
ds = xr.open_dataset("/work/mh0010/gridsat_b1/1990/GRIDSAT-B1.1990.04.14.15.v02r01.nc")

# %%
ds['irwin_cdr'].isel(time=0).plot.imshow()
# %% 
