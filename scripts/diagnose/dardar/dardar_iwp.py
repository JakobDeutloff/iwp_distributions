# %%
import xarray as xr
import glob
from tqdm import tqdm
import numpy as np
import sys 
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

# %%
year = sys.argv[1]
batch_idx = int(sys.argv[2])

# %% 
files = glob.glob(f"/work/bm1183/m301049/dardarv3.10/{year}/*/DARDAR*.nc")
files.sort()
batch_number = 10 
batch_size = int(np.ceil((len(files)/batch_number)))
batch = [files[i:i+batch_size] for i in range(0, len(files), batch_size)][batch_idx]

# %%
def open_and_check(file):
    try:
        ds = xr.open_dataset(file)
        time_flag = ds.time[0].dtype == '<M8[ns]'
    except Exception as e:
        print(f"Error opening {file}: {e}")
        return None
    if not time_flag:
        print(f"Time variable in {file} is not in datetime64 format.")
        return None
    else:
        ds = ds.drop_attrs()
        return ds
# %% 
datasets = []
print(f"Processing year: {year} batch {str(batch_idx)}")
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(open_and_check, f) for f in batch]
    for future in tqdm(as_completed(futures), total=len(futures)):
        datasets.append(future.result())

# drop None datasets
datasets = [ds for ds in datasets if ds is not None]
# %% check dimensions
ref_dims = set(datasets[0].dims.keys())
ref_vars = set(datasets[0].data_vars.keys())

for i, ds in enumerate(datasets[1:], 1):
    ds_dims = set(ds.dims.keys())
    ds_vars = set(ds.data_vars.keys())
    if ds_dims != ref_dims:
        print(f"Dataset {i} has different dimension names: {ds_dims}")
        datasets.pop(i)
    if ds_vars != ref_vars:
        print(f"Dataset {i} has different variable names: {ds_vars}")
        datasets.pop(i)

# %% concatenate
print('Concatenating')
ds = xr.concat(datasets, dim='time')

# %% calculate iwp 
print ('Calculating IWP')
dz = np.abs(ds['height'].diff('height').values)
dz = np.append(dz, dz[-1])
dz = xr.DataArray(dz, dims=['height'], coords={'height': ds['height']})
iwp = (ds['iwc'] * dz).sum('height')
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
ds_iwp.to_netcdf(f"/work/bm1183/m301049/dardarv3.10/{year}/iwp_dardar_{year}_{batch_idx}.nc")
