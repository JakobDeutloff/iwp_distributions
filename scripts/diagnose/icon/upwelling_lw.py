# %%
import intake
from src.healpix_functions import attach_coords, sel_region
import numpy as np
import xarray as xr

# %%
path = (
    "/work/bm1235/k203123/nextgems_prefinal/experiments/ngc4008a/outdata/ngc4008a.yaml"
)
cat = intake.open_catalog(path)
ds_icon_inst = (
    cat.ngc4008a(chunks="auto", zoom=9, time="PT15M").to_dask().pipe(attach_coords)
)
ds = sel_region(ds_icon_inst.sel(time="2021-07-02"), -30, 30, 0, 360)
# %% attach local time
local_time = (ds["time"].dt.hour + (ds["time"].dt.minute / 60) + (ds["lon"] / 15)) % 24
ds = ds.assign(
    {
        "local_time": (
            ("time", "cell"),
            local_time.data,
        ),
    }
)

# %% calculate condensate as measure for clearsky
ds = ds.assign(
    condensate=(
        ("time", "cell"),
        (ds["cli"].sum("level_full") + ds["clw"].sum("level_full")).data,
    )
)

# %% load ds
ds_sel = ds[["rlut", "condensate", "local_time"]].load()

# %% get clearsky values
lt_bins = np.arange(0, 24.2, 0.2)
lt_points = (lt_bins[:-1] + lt_bins[1:]) / 2
rlut_cs = (
    ds_sel["rlut"]
    .where(ds_sel["condensate"] < ds_sel["condensate"].quantile(0.3))
    .groupby_bins(ds_sel["local_time"], lt_bins)
    .mean()
)
rlut_cs = rlut_cs.rename({"local_time_bins": "local_time"})
rlut_cs["local_time"] = lt_points

# %% save rlut
rlut_cs.to_netcdf("/work/bu1562/m301049/iwp_dists/publication/fluxes/rlut_cs.nc")
