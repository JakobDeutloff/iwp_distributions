# %%
import xarray as xr 
import numpy as np
from src.helper_functions import shift_longitudes

# %% 
temp_all = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m.nc")

# %%
mask = (
    xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")
    .load()
    .pipe(shift_longitudes, lon_name="lon")
)
mask = mask.sel(lon=temp_all.longitude, lat=temp_all.latitude, method='nearest')

# %% calculate mean tropical temp
temp_trop = (
    temp_all.sel(latitude=slice(30, -30))
    * np.cos(np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude))
).mean(["latitude", "longitude"]) / np.cos(
    np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude)
).mean(
    "latitude"
)
temp_trop = temp_trop.rename({'valid_time': 'time'})
temp_trop.to_netcdf("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc")

# %% calculate mean sea tropical temp 
temp_sea = (
    temp_all.where(mask).sel(latitude=slice(30, -30))
    * np.cos(np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude))
).mean(["latitude", "longitude"]) / np.cos(
    np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude)
).mean(
    "latitude"
)
temp_sea = temp_sea.rename({'valid_time': 'time'})
temp_sea.to_netcdf("/work/bm1183/m301049/era5/monthly/t2m_tropics_sea.nc")


# %% calculate mean land tropical temp
temp_land = (
    temp_all.where(mask==0).sel(latitude=slice(30, -30))
    * np.cos(np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude))
).mean(["latitude", "longitude"]) / np.cos(
    np.deg2rad(temp_all.sel(latitude=slice(30, -30)).latitude)
).mean(
    "latitude"
)
temp_land = temp_land.rename({'valid_time': 'time'})
temp_land.to_netcdf("/work/bm1183/m301049/era5/monthly/t2m_tropics_land.nc")

# %%
