# %%
import xarray as xr
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import sys
import dask
import gc
import glob


warnings.filterwarnings("ignore", category=FutureWarning, module="xarray.core.concat")


# %%
def process_dir(root, dir):
    #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    ds = xr.open_mfdataset(os.path.join(root, dir, "DARDAR*.nc")).load()
    dz = np.abs(ds["height"].diff("height")).values
    dz = np.append(dz, dz[-1])
    dz = xr.DataArray(dz, dims=["height"], coords={"height": ds["height"]})
    iwp = (ds["iwc"] * dz).sum("height")
    ds_iwp = ds[
        [
            "latitude",
            "longitude",
            "land_water_mask",
            "day_night_flag",
            "sea_surface_temperature",
        ]]
    ds_iwp['iwp'] = iwp
    ds_iwp['iwp'].attrs = {
        'long_name': 'Ice Water Path',
        'units': 'kg/m^2',
        'standard_name': 'ice_water_path'
    }
    #ds_iwp = ds_iwp.sortby("time").drop_duplicates("time")
    ds_iwp.to_netcdf(os.path.join(root, dir, f"iwp_{dir}.nc"))
    # clear memory
    ds.close()
    ds_iwp.close()
    del ds, ds_iwp
    gc.collect()


# %% 
year = sys.argv[1]
root = f"/work/bm1183/m301049/dardarv3.10/{year}/"
dirs = glob.os.listdir(root)
for dir in tqdm(dirs):
    process_dir(root, dir)

# %%
