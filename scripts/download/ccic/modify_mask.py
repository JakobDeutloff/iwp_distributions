# %%
import xarray as xr
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm

# %%
mask = xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")


# %%
# Create buffer zone around land (set ocean within 100 km of land to 0)
def create_land_buffer(mask_data, lats, lons, buffer_km=100):
    """
    Set ocean points within buffer_km of land to 0.

    Parameters:
    -----------
    mask_data : array
        Land-sea mask (0=land, 1=ocean)
    lats : array
        Latitude values (degrees)
    lons : array
        Longitude values (degrees)
    buffer_km : float
        Buffer distance in kilometers

    Returns:
    --------
    modified_mask : array
        Modified mask with buffer zone
    """
    # Calculate grid spacing in degrees
    dlat = np.abs(np.mean(np.diff(lats)))
    dlon = np.abs(np.mean(np.diff(lons)))

    # Calculate approximate grid spacing in km at different latitudes
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(lat)
    lat_km_per_deg = 111.0

    # Use median latitude for approximate conversion
    med_lat = np.median(lats)
    lon_km_per_deg = 111.0 * np.cos(np.radians(med_lat))

    # Calculate number of grid cells for the buffer
    n_lat_cells = int(np.ceil(buffer_km / (dlat * lat_km_per_deg)))
    n_lon_cells = int(np.ceil(buffer_km / (dlon * lon_km_per_deg)))

    # Use maximum to be conservative
    n_cells = max(n_lat_cells, n_lon_cells)

    # Find land points (mask == 0)
    land_mask = mask_data.values == 0

    # Use binary dilation to expand land areas
    structure = ndimage.generate_binary_structure(2, 2)
    dilated_land = ndimage.binary_dilation(
        land_mask, structure=structure, iterations=n_cells
    )

    # Set ocean points within buffer to 0
    modified_mask = np.copy(mask_data.values)
    modified_mask[dilated_land & (mask_data.values == 1)] = 0

    # build back xarray DataArray
    modified_mask = xr.DataArray(
        modified_mask,
        coords=mask_data.coords,
        dims=mask_data.dims,
        attrs=mask_data.attrs,
    )

    return modified_mask


def remove_small_islands(mask_data, size_threshold=50):
    """
    Remove small islands (land areas smaller than size_threshold pixels).

    Parameters:
    -----------
    mask_data : array
        Land-sea mask (0=land, 1=ocean)
    size_threshold : int
        Minimum size of land areas to keep (in pixels)

    Returns:
    --------
    cleaned_mask : array
        Mask with small islands removed
    """
    # Find land points (mask == 0)
    land_mask = mask_data == 0

    # Label connected land regions
    labeled_array, num_features = ndimage.label(land_mask)

    # Count pixels in each region (vectorized)
    region_sizes = np.bincount(labeled_array.ravel())

    # Create mask of regions to remove (small islands)
    small_regions_mask = region_sizes < size_threshold
    small_regions_mask[0] = False  # Don't remove background (label 0)

    # Remove small regions by setting them to ocean (1)
    cleaned_mask = np.copy(mask_data.values)
    cleaned_mask[small_regions_mask[labeled_array]] = 1

    # build back xarray DataArray
    cleaned_mask = xr.DataArray(
        cleaned_mask,
        coords=mask_data.coords,
        dims=mask_data.dims,
        attrs=mask_data.attrs,
    )

    return cleaned_mask


# %% Apply small island removal
mask_big_cluster = remove_small_islands(mask, size_threshold=2000)
# %% Apply the buffer
modified_mask = create_land_buffer(
    mask_big_cluster, mask_big_cluster.lat.values, mask_big_cluster.lon.values, buffer_km=1000
)

# %% coarsen masks for plotting
coarsen_factor = 10
modified_mask_coarse = modified_mask.coarsen(
    lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
).max()
mask_coarse = mask.coarsen(
    lat=coarsen_factor, lon=coarsen_factor, boundary="trim"
).max()

# %% make a test plot
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
ax.contour(
    mask_coarse.lon,
    mask_coarse.lat,
    mask_coarse,
    levels=[0.5],
    colors="blue",
    linewidths=1,
    transform=ccrs.PlateCarree(),
)
ax.contour(
    modified_mask_coarse.lon,
    modified_mask_coarse.lat,
    modified_mask_coarse,
    levels=[0.5],
    colors="red",
    linewidths=1,
    transform=ccrs.PlateCarree(),
)
# %% calculate fraction of tropical oceans
trop_ocean = (mask.sel(lat=slice(-30, 30)) == 1).mean().item()
trop_ocean_modified = (modified_mask.sel(lat=slice(-30, 30)) == 1).mean().item()
print(f"Original tropical ocean fraction: {trop_ocean:.3f}")
print(f"Modified tropical ocean fraction: {trop_ocean_modified:.3f}")

# %% save modified mask 
modified_mask.to_netcdf("/work/bm1183/m301049/orcestra/modified_sea_land_mask_1000.nc")

# %%
