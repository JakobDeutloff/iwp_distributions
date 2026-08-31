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
        "/work/bu1562/m301049/GPM_MERGIR/merg_2022010122*.nc4", engine="netcdf4"
    )
    .sel(lat=slice(-30, 30))
    .load()
)
# %%
s3 = s3fs.S3FileSystem(anon=True)
prefix = f"chalmerscloudiceclimatology/record/cpcir/2022/ccic_cpcir_2022010122*"
files = s3.glob(prefix)
ds = xr.open_zarr(s3.get_mapper(files[0]))
ds = ds.sel(latitude=slice(30, -30)).load()
iwp = ds["tiwp"]
#  reverse iwp latitude to match bt
iwp = iwp[:, ::-1, :]
iwp = iwp.fillna(0)
# %%
icon = (
    xr.open_dataset(
        f"/work/bu1562/m301049/icon-mpim/experiments/jed0011/jed0011_atm_2d_19790701T000040Z.15356915.nc",
        chunks={"time": 1, "ncells": -1},
    ).pipe(merge_grid)
).isel(time=0)

icon = to_healpix(icon)
iwp_icon = (
    icon["clivi"] + icon["qsvi"] + icon["qgvi"] + icon["qrvi"] + icon["cllvi"]
).load()


# %%
background = (1, 1, 1)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#110734"])
projection = ccrs.PlateCarree()

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
fig.savefig("plots/diurnal_cycle/talk/icon_iwp.png", bbox_inches="tight", dpi=400)

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
)
ax.axis("off")
fig.savefig("plots/diurnal_cycle/talk/ccic_iwp.png", bbox_inches="tight", dpi=400)

# %% plot bts
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": projection})
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # Turn off axes BEFORE plotting
ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
bts = bts.coarsen(lon=2, lat=2, boundary="trim").mean()
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
fig.savefig("plots/diurnal_cycle/talk/bts_tb.png", bbox_inches="tight", dpi=400)

# %% check for complete bt data
null = bts.sel(lat=slice(-30, 30)).isnull().sum(["lat", "lon"])
null["Tb"].plot()
null["Tb"].idxmin("time")
# %%
