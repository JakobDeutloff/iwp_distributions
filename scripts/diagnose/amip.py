# %%
import xarray as xr 
import intake 
import intake_esm
# %% 
cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")
time_slice = slice('1980-01-01','1993-12-31')
ds_4k = cat['ICON.C5.AMIP_P4K'](zoom=8, time="PT3H", chunks="auto").to_dask().sel(time=time_slice)
ds_cont = cat['ICON.C5.AMIP_CNTL'](zoom=8, time="PT3H", chunks="auto").to_dask().sel(time=time_slice)

# %% 
iwp = (ds_cont['clivi'].isel(time=-1) + ds_cont['qgvi'].isel(time=-1) + ds_cont['qsvi'].isel(time=-1)).compute()

# %%
