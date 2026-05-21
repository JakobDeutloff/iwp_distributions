# %%
import matplotlib.pyplot as plt
import xarray as xr
from src.grid_helpers import merge_grid, to_healpix
import ccic
import cartopy.crs as ccrs
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import easygems.healpix as egh
import numpy as np
from src.grid_helpers import to_healpix, merge_grid
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import s3fs

# %% load aquaplanet data and iwp and bt data
bts = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/GPM_MERGIR/merg_2022010122*.nc4", engine="netcdf4"
    )
    .sel(lat=slice(-30, 30))
    .load()
)

# %% ccic
s3 = s3fs.S3FileSystem(anon=True)
prefix = f"chalmerscloudiceclimatology/record/cpcir/2022/ccic_cpcir_2022010122*"
files = s3.glob(prefix)
ds = xr.open_zarr(s3.get_mapper(files[0]))
ds = ds.sel(latitude=slice(30, -30)).load()
iwp = ds["tiwp"]
#  reverse iwp latitude to match bt
iwp = iwp[:, ::-1, :]
iwp = iwp.fillna(0)
# %% icon
icon = (
    xr.open_dataset(
        f"/work/bu1562/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790701T000040Z.15356915.nc",
        chunks={"time": 1, "ncells": -1},
    ).pipe(merge_grid)
).isel(time=0)

icon = to_healpix(icon)
iwp_icon = (
    icon["clivi"] + icon["qsvi"] + icon["qgvi"]
).load()

# %% land sea mask 
mask = xr.open_dataarray("/work/bm1183/m301049/orcestra/sea_land_mask.nc")
mask = mask.sel(lat=slice(-30, 30)).load()
mask = mask.sel(
        lon=ds["longitude"], lat=ds["latitude"], method="nearest"
    ).drop_vars(["lon", "lat"])
# %% 
background = (1, 1, 1)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#110734"])
projection = ccrs.Mollweide()

# %% plot icon
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": projection})
fig.set_dpi(400)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # Turn off axes BEFORE plotting
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
_, _, nx, ny = np.array(ax.bbox.bounds, dtype=int)

xlims = ax.get_xlim()
ylims = ax.get_ylim()
im = egh.healpix_resample(
    iwp_icon.values, xlims, ylims, nx, ny, ax.projection, "nearest", nest=True
)
im = im.fillna(0)
vmin, vmax = 6e-2, 1e1
norm = LogNorm(vmin=vmin, vmax=vmax)

ax.imshow(im, extent=xlims + ylims, origin="lower", cmap=white_cmap, norm=norm)
ax.axis("off")
fig.savefig('plots/diurnal_cycle/talk/icon_iwp.png', bbox_inches='tight', dpi=400)

# %% plot ccic
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": projection})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # Turn off axes BEFORE plotting
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
ax.pcolormesh(
    iwp["longitude"],
    iwp["latitude"],
    iwp.isel(time=0),
    cmap=white_cmap,
    norm=LogNorm(6e-2, 1e1),
    rasterized=True,
    transform=ccrs.PlateCarree(),
)
ax.axis("off")
fig.savefig('plots/diurnal_cycle/talk/ccic_iwp.png', bbox_inches='tight', dpi=400)

# %% plot bts 
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": projection})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # Turn off axes BEFORE plotting
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
bts=bts.coarsen(lon=2, lat=2, boundary='trim').mean()
ax.pcolormesh(
    bts["lon"],
    bts["lat"],
    bts["Tb"].isel(time=0),
    cmap="inferno",
    rasterized=True,
    vmin=200,
    vmax=290,
)
ax.axis("off")
fig.savefig('plots/diurnal_cycle/talk/bts_tb.png', bbox_inches='tight', dpi=400)

# %% check for complete bt data 
null = bts.sel(lat=slice(-30, 30)).isnull().sum(['lat', 'lon'])
null['Tb'].plot()
null['Tb'].idxmin('time')
# %% plot ccic and icon in one plot 
background = (1, 1, 1)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#000000"])
projection = ccrs.PlateCarree()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), subplot_kw={"projection": projection}, sharex=False, sharey=True)
fig.set_dpi(400)
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")  # Turn off axes BEFORE plotting
    ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    ax.set_yticks([-30, 0, 30], crs=ccrs.PlateCarree())
    ax.set_yticklabels(['30°S', '0°', '30°N'])


# plot icon
xlims = axes[0].get_xlim()
ylims = axes[0].get_ylim()
im = egh.healpix_resample(
    iwp_icon.values, xlims, ylims, nx, ny, axes[0].projection, "nearest", nest=True
)
im = im.fillna(0)
vmin, vmax = 1e-3, 10
norm = LogNorm(vmin=vmin, vmax=vmax)
axes[0].imshow(im, extent=xlims + ylims, origin="lower", cmap=white_cmap, norm=norm)

# plot ccic
im = axes[1].pcolormesh(
    iwp["longitude"],
    iwp["latitude"],
    iwp.isel(time=0),
    cmap=white_cmap,
    norm=LogNorm(1e-3, 10),
    rasterized=True,
    transform=ccrs.PlateCarree(),
)
cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, extend='both', aspect=40)
cb.set_label('$I$ / kg m$^{-2}$')
# set labels at 30N and 30S and -180 and 180 of axes 
axes[1].set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
axes[0].set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
axes[1].set_xticklabels(['180°W', '90°W', '0°', '90°E', '180°E'])
axes[0].set_xticklabels([])
axes[0].set_title('ICON Aquaplanet')
axes[1].set_title('CCIC')

fig.savefig('plots/thesis/icon_ccic_iwp.pdf', bbox_inches='tight', dpi=400)



# %% make plot for thesis cover
# %% plot ccic and icon in one plot 

cmap_white = LinearSegmentedColormap.from_list(
    "white_alpha",
    [
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 1.0), 
    ],
)
cmap_icon = LinearSegmentedColormap.from_list("white", ["#020640", (1, 1, 1, 1)])
cmap_ccic = LinearSegmentedColormap.from_list("white", [(1, 1, 1, 1), "#023201"])
projection = ccrs.PlateCarree()

fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), subplot_kw={"projection": projection})
fig.set_dpi(400)
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# plot icon
cmap_white = cmap_white.copy()
cmap_white.set_bad((1, 1, 1, 0))    # invalid (e.g. 0 in LogNorm) -> transparent
cmap_white.set_under((1, 1, 1, 0))  # below vmin -> transparent
axes[0].set_facecolor("#02022E") 
xlims = axes[0].get_xlim()
ylims = axes[0].get_ylim()
im = egh.healpix_resample(
    iwp_icon.values, xlims, ylims, nx, ny, axes[0].projection, "nearest", nest=True
)
im = im.fillna(0)
vmin, vmax = 1e-3, 10
norm = LogNorm(vmin=vmin, vmax=vmax)
axes[0].imshow(im, extent=xlims + ylims, origin="lower", cmap=cmap_white, norm=norm)


# plot ccic
axes[1].pcolormesh(
    mask["longitude"],
    mask["latitude"],
    mask,
    cmap=LinearSegmentedColormap.from_list("white", ["#011E00", "#02022E",]), # two colors, one for land and one for sea
    rasterized=True,
    transform=ccrs.PlateCarree(),
)
im = axes[1].pcolormesh(
    iwp["longitude"],
    iwp["latitude"],
    iwp.isel(time=0),
    cmap=cmap_white,
    norm=norm,
    rasterized=True,
    transform=ccrs.PlateCarree(),
)

fig.savefig('plots/thesis/icon_ccic_iwp_cover.pdf', dpi=400, bbox_inches='tight')
# %% next try


cmap_icon = LinearSegmentedColormap.from_list("white", [(1, 1, 1, 1), "#020640",])
cmap_ccic = LinearSegmentedColormap.from_list("white", [(1, 1, 1, 1), "#023201"])
projection = ccrs.PlateCarree()

fig, axes = plt.subplots(2, 1, figsize=(15, 10), subplot_kw={"projection": projection})
fig.set_dpi(400)
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_extent([-90, 90, -30, 30], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# plot icon
xlims = axes[0].get_xlim()
ylims = axes[0].get_ylim()
im = egh.healpix_resample(
    iwp_icon.values, xlims, ylims, nx, ny, axes[0].projection, "nearest", nest=True
)
im = im.fillna(0)
vmin, vmax = 1e-3, 10
norm = LogNorm(vmin=vmin, vmax=vmax)
axes[0].imshow(im, extent=xlims + ylims, origin="lower", cmap=cmap_icon, norm=norm)


# plot ccic
im = axes[1].pcolormesh(
    iwp["longitude"],
    iwp["latitude"],
    iwp.isel(time=0),
    cmap=cmap_ccic,
    norm=norm,
    rasterized=True,
    transform=ccrs.PlateCarree(),
)

fig.savefig('plots/thesis/icon_ccic_iwp_cover.pdf', dpi=400, bbox_inches='tight')
# %%
