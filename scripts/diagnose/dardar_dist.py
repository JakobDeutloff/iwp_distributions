# %%
import xarray as xr
import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

# %%
year = '2006'
path = f"/work/bm1183/m301049/dardarv3.10/{year}/"
files = []
roots = []
for root, dirs, file_names in os.walk(path):
    roots.append(root)
    for file in file_names:
        if file.startswith(f"iwp_{year}") and file.endswith(".nc"):
            files.append(xr.open_dataset(os.path.join(root, file)))

# %% concatenate files and throw away duplicates
ds = xr.concat(files, dim='time')

# %%
ds = xr.open_mfdataset(f"{path}/*/iwp_2006*.nc")
# %%
