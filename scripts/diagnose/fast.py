import xarray as xr
from dask.diagnostics import ProgressBar

chunks = {"latitude": 180, "longitude": 360, "hybrid": 10}
path = "/work/bm1183/m301049/era5/monthly"

t = xr.open_dataarray(f"{path}/130_an_latlon.grb", engine="cfgrib", chunks=chunks)
p = xr.open_dataarray(f"{path}/pressure_latlon.nc", chunks=chunks)
sw = xr.open_dataarray(f"{path}/235003_fc_latlon.grb", engine="cfgrib", chunks=chunks)
lw = xr.open_dataarray(f"{path}/235004_fc_latlon.grb", engine="cfgrib", chunks=chunks)

R, cp = 8.314, 29.07
seconds_of_day = 24 * 60 * 60

rad_tendency = (-(sw + lw) * seconds_of_day).chunk(chunks)
dp = p.differentiate("hybrid")
dt = t.differentiate("hybrid")
dt_dp = (dt / dp).persist()
stability = ((t / p) * (R / cp) - dt_dp).persist()
w_r = (rad_tendency / stability).chunk(chunks)
conv = (w_r.differentiate("hybrid") / dp).chunk(chunks)

with ProgressBar():
    for name, arr in [
        ("rad_tendency", rad_tendency),
        ("lapse_rate", dt_dp),
        ("stability", stability),
        ("subsidence", w_r),
        ("convergence", conv),
    ]:
        print(f"Saving {name}")
        arr.compute().to_netcdf(f"{path}/{name}.nc", engine="h5netcdf")