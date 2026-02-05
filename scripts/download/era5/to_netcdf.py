# %%
import xarray as xr

path = "/work/bm1183/m301049/era5/monthly"
files = {
    "t": "130_an_latlon.grb",
    "sw": "235003_fc_latlon.grb",
    "lw": "235004_fc_latlon.grb",
}
# %%
for var, filename in files.items():
    print(f"Processing {var}...")
    ds = xr.open_dataarray(f"{path}/{filename}", engine="cfgrib", chunks={})
    # Set chunk sizes for each dimension (adjust as needed)
    encoding = {ds.name: {"chunksizes": [ds.sizes[dim] if dim != "longitude" else 10 for dim in ds.dims]}}
    ds.to_netcdf(f"{path}/{var}.nc", encoding=encoding)
    ds.close()

# %% chunk pressure 
p =  xr.open_dataarray(f"{path}/pressure_latlon.nc", chunks={})
p.name = "pressure"
encoding = {p.name: {"chunksizes": [p.sizes[dim] if dim != "longitude" else 10 for dim in p.dims]}}
p.to_netcdf(f"{path}/p.nc", encoding=encoding)

# %%
