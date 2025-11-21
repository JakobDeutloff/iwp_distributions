# %%
import xarray as xr
import glob
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
from scipy.interpolate import griddata
from functools import partial
from dask.diagnostics import ProgressBar
# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords").chunk({'time': 1, 'values':-1})

# %% interpolate to regular grid
lat = np.unique(ds['latitude'])
lat.sort()
lon = np.arange(0, 360.25, 0.25)
grid_lon, grid_lat = np.meshgrid(lon, lat)
# %%
# Interpolate
def interp_time(t):
    return griddata(
        (ds['latitude'].values, ds['longitude'].values),
        ds['t2m'].isel(time=t).values,
        (grid_lat, grid_lon),
        method='linear'
    )

gridded_data = np.zeros((len(ds['time']), len(lat), len(lon)))

with ProcessPoolExecutor(max_workers=64) as executor:
    futures = {executor.submit(interp_time, t): t for t in range(len(ds['time']))}
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        t = futures[future]
        gridded_data[t, :, :] = future.result()

ds_grid = xr.DataArray(
    gridded_data,
    coords={"time": ds['time'], "latitude": lat, "longitude": lon},
    dims=["time", "latitude", "longitude"],
    name="t2m",
    attrs={"units": "K", "long_name": "2m Temperature"},
)
# %%
def interp_to_grid(t2m_1d, lats, lons, grid_lat, grid_lon):
    return griddata(
        (lats, lons),
        t2m_1d,
        (grid_lat, grid_lon),
        method='linear'
    )

# Prepare your grid
lat = np.arange(-90, 90.25, 0.25)
lon = np.arange(0, 360.25, 0.25)
grid_lon, grid_lat = np.meshgrid(lon, lat)

# Use functools.partial to fix the static arguments
interp_func = partial(interp_to_grid, lats=ds['latitude'].values, lons=ds['longitude'].values, grid_lat=grid_lat, grid_lon=grid_lon)

result = xr.apply_ufunc(
    interp_func,
    ds['t2m'],
    input_core_dims=[['values']],
    output_core_dims=[['latitude', 'longitude']],
    dask_gufunc_kwargs={'output_sizes': {'latitude': len(lat), 'longitude': len(lon)}},
    vectorize=True,
    dask='parallelized',
    output_dtypes=[ds['t2m'].dtype],
)
result = result.assign_coords(latitude=lat, longitude=lon)
with ProgressBar():
    result = result.compute()
# %%
