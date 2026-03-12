import xarray as xr
import healpy as hp
from dask.diagnostics import ProgressBar
import easygems.remap as egr
import numpy as np
import os

def get_grid(ds):
    uri = ds.grid_file_uri
    downloads_prefix = "http://icon-downloads.mpimet.mpg.de/grids/public/"
    if uri.startswith(downloads_prefix):
        local_grid_path = os.path.join(
            "/pool/data/ICON/grids/public", uri[len(downloads_prefix) :]
        )
    else:
        raise NotImplementedError(f"no idea about how to get {uri}")
    return xr.open_dataset(local_grid_path)

def merge_grid(data_ds):
    grid = get_grid(data_ds)
    ds = xr.merge([data_ds, grid[list(grid.coords)].rename({"cell": "ncells"})])
    ds = ds.assign(clon=np.degrees(ds.clon), clat=np.degrees(ds.clat))
    ds = ds.assign_coords(ncells=np.arange(ds.sizes["ncells"]))
    return ds

def to_healpix(ds, save_path=None, zoom=10):
    """
    Regrid ds from ICON R2B9 grid to healpix grid
    """

    # load weights
    weights = xr.open_dataset("/home/m/m301049/aqua_processing/data/weights_z10.nc")
    order = zoom
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    # remap ds
    print("remapping")
    ds_remap = xr.apply_ufunc(
        egr.apply_weights,
        ds,
        kwargs=weights,
        keep_attrs=True,
        input_core_dims=[["ncells"]],
        output_core_dims=[["cell"]],
        output_dtypes=["f4"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"cell": npix},
        },
    )
    # attach grid metadata
    ds_remap["crs"] = xr.DataArray(
        name="crs",
        data=[],
        dims="crs",
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": 2**zoom,
            "healpix_order": "nest",
        },
    )

    # save dataset to nc
    if save_path is not None:
        print("saving")
        with ProgressBar():
            ds_remap.to_netcdf(save_path)
    else:
        return ds_remap
