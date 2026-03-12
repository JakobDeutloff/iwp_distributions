import numpy as np
import healpy as hp
import dask


def get_nest(ds):
    return ds.crs.healpix_order == "nest"


def get_nside(ds):
    return ds.crs.healpix_nside


def attach_coords(ds):
    lons, lats = hp.pix2ang(
        get_nside(ds), np.arange(ds.dims["cell"]), nest=get_nest(ds), lonlat=True
    )
    return ds.assign_coords(
        lat=(("cell",), lats, {"units": "degree_north"}),
        lon=(("cell",), lons, {"units": "degree_east"}),
    )


def sel_region(ds, lat_min, lat_max, lon_min, lon_max):
    if lon_min < lon_max:
        bool_idx = ((ds["lon"] <= lon_max) & (ds["lon"] >= lon_min)) & (
            (ds["lat"] <= lat_max) & (ds["lat"] >= lat_min)
        )
    else:
        bool_idx = ((ds["lon"] <= lon_max) | (ds["lon"] >= lon_min)) & (
            (ds["lat"] <= lat_max) & (ds["lat"] >= lat_min)
        )
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):  # Takes too long to avoid large chunks 
        ds_region = ds.isel(cell=bool_idx)
    return ds_region


def sel_section(ds, startpoint, endpoint, section_direction=None):
    # get aproximate resolution of dataset and reduce it by 10% to make sure to get all cells
    res = hp.nside2resol(get_nside(ds), arcmin=True) / 60
    res = res - 0.1 * res

    # determine the number od steps from start to end point - set by the higher difference in lat or lon
    steps = np.max(
        [
            np.abs(startpoint["lon"] - endpoint["lon"] / res),
            np.abs(startpoint["lat"] - endpoint["lat"] / res),
        ]
    )

    # create lat and lon arrays
    latitudes = np.linspace(startpoint["lat"], endpoint["lat"], int(np.ceil(steps)))
    longitudes = np.linspace(startpoint["lon"], endpoint["lon"], int(np.ceil(steps)))

    # Select cells of the profile and remove duplicates
    cells_selected = np.unique(
        hp.ang2pix(
            nside=get_nside(ds),
            theta=longitudes,
            phi=latitudes,
            nest=get_nest(ds),
            lonlat=True,
        )
    )
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds_section = ds.sel(cell=cells_selected)
    
    if section_direction == 'lat':
        ds_section = ds_section.sortby('lat')
    elif section_direction == 'lon':
        ds_section = ds_section.sortby('lon')
    else:
        pass

    return ds_section
