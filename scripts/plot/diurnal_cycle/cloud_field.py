# %%
import xarray as xr
import matplotlib.pyplot as plt
import ccic
from matplotlib.colors import LinearSegmentedColormap, LogNorm


# %%
bts = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/GPM_MERGIR/merg_2008010901*.nc4", engine="netcdf4"
    )
    .sel(lat=slice(-30, 30))
    .load()
)
# %%
iwp = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/ccic/raw/ccic_cpcir_2008010901*.zarr", engine="zarr"
    )
    .sel(latitude=slice(30, -30))
    .load()
)["tiwp"]

#  reverse iwp latitude to match bt
iwp = iwp[:, ::-1, :]
iwp = iwp.fillna(0)


# %% plot bt and iwp snapshots
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
# make a logarithmic for iwp from dark blue to white
iwp_cmap = LinearSegmentedColormap.from_list("iwp_cmap", ["#000000", "#fcfcfc"])
lat_range = slice(-5, 1)
lon_range = slice(22, 29)
# lat_range = slice(0, 8)
# lon_range = slice(100, 115)
axes[1].set_facecolor("#000000")

im0 = axes[0].pcolormesh(
    bts["lon"].sel(lon=lon_range),
    bts["lat"].sel(lat=lat_range),
    bts["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    cmap="inferno",
    rasterized=True,
    vmin=200,
    vmax=290,
)
ct0 = axes[0].contour(
    bts["lon"].sel(lon=lon_range),
    bts["lat"].sel(lat=lat_range),
    bts["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    levels=[231, 261],
    colors="white",
    linewidths=2,
    linestyles=["solid", "dotted"],
)
im1 = axes[1].pcolormesh(
    iwp["longitude"].sel(longitude=lon_range),
    iwp["latitude"].sel(latitude=lat_range),
    iwp.sel(latitude=lat_range, longitude=lon_range).isel(time=0),
    cmap=iwp_cmap,
    norm=LogNorm(1e-3, 1e1),
    rasterized=True,
)
ct1 = axes[1].contour(
    iwp["longitude"].sel(longitude=lon_range),
    iwp["latitude"].sel(latitude=lat_range),
    iwp.sel(latitude=lat_range, longitude=lon_range).isel(time=0),
    levels=[1e-1, 1],
    colors="black",
    linewidths=2,
    linestyles=["dotted", "solid"],
)
for ax in axes:
    ax.set_xlabel("Longitude / °E")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Latitude / °N")

# make legends
leg = axes[0].legend(
    [
        plt.Line2D([], [], color="white", linestyle=":"),
        plt.Line2D([], [], color="white", linestyle="-"),
    ],
    ["$T_{b} = 261$ K", "$T_{b} = 231$ K"],
    loc="upper right",
    facecolor="black",
    edgecolor="black",
)
for text in leg.get_texts():
    text.set_color("white")

leg = axes[1].legend(
    [
        plt.Line2D([], [], color="black", linestyle=":"),
        plt.Line2D([], [], color="black", linestyle="-"),
    ],
    ["$I$ = 0.1 kg m$^{-2}$", "$I$ = 1 kg m$^{-2}$"],
    loc="upper right",
    facecolor="white",
    edgecolor="black",
)

fig.colorbar(
    im0,
    ax=axes[0],
    label="$T_{b}$ / K",
    orientation="horizontal",
    extend="both",
    pad=0.1,
)
fig.colorbar(
    im1,
    ax=axes[1],
    label="$I$ / kg m$^{-2}$",
    orientation="horizontal",
    extend="both",
    pad=0.1,
)
# add letter 
for ax, letter, color in zip(axes, ["a", "b"], ['black', 'white']):
    ax.text(
        0.02,
        0.93,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=color,
    )
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/publication/bt_iwp_snapshot.pdf")


# %% plot for talk 
fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
background = (1, 1, 1)
white_cmap = LinearSegmentedColormap.from_list("white", [background, "#110734"])
lat_range = slice(-5, 1)
lon_range = slice(22, 29)
axes[1].set_facecolor('white')

im0 = axes[1].pcolormesh(
    bts["lon"].sel(lon=lon_range),
    bts["lat"].sel(lat=lat_range),
    bts["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    cmap="inferno",
    rasterized=True,
    vmin=200,
    vmax=290,
)
im1 = axes[0].pcolormesh(
    iwp["longitude"].sel(longitude=lon_range),
    iwp["latitude"].sel(latitude=lat_range),
    iwp.sel(latitude=lat_range, longitude=lon_range).isel(time=0),
    cmap=white_cmap,
    norm=LogNorm(1e-3, 1e1),
    rasterized=True,
)
ct0 = axes[1].contour(
    bts["lon"].sel(lon=lon_range),
    bts["lat"].sel(lat=lat_range),
    bts["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    levels=[231],
    colors="white",
    linewidths=2,
    linestyles=["solid"],
)
ct1 = axes[0].contour(
    iwp["longitude"].sel(longitude=lon_range),
    iwp["latitude"].sel(latitude=lat_range),
    iwp.sel(latitude=lat_range, longitude=lon_range).isel(time=0),
    levels=[1],
    colors="black",
    linewidths=2,
    linestyles=["solid"],
)
for ax in axes:
    ax.set_xlabel("Longitude / °E")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Latitude / °N")

fig.colorbar(
    im0,
    ax=axes[1],
    label="$T_{b}$ / K",
    orientation="horizontal",
    extend="both",
    pad=0.1,
)
fig.colorbar(
    im1,
    ax=axes[0],
    label="$I$ / kg m$^{-2}$",
    orientation="horizontal",

    extend="both",
    pad=0.1,
)

fig.tight_layout()
fig.savefig("plots/diurnal_cycle/talk/bt_iwp_cont.png", dpi=300)


# %% fine grained vs coarse grained 
bts_coarse = bts.coarsen(lat=26, lon=26, boundary="trim").mean()
white_cmap = LinearSegmentedColormap.from_list("white", [(1, 1, 1), (0, 0, 0)])
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im_fine = axes[0].pcolormesh(
    bts["lon"].sel(lon=lon_range),
    bts["lat"].sel(lat=lat_range),
    bts["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    cmap=white_cmap,
    rasterized=True,
    vmin=200,
    vmax=290,
)

im_coarse = axes[1].pcolormesh(
    bts_coarse["lon"].sel(lon=lon_range),
    bts_coarse["lat"].sel(lat=lat_range),
    bts_coarse["Tb"].sel(lat=lat_range, lon=lon_range).isel(time=0),
    cmap=white_cmap,
    rasterized=True,
    vmin=200,
    vmax=290,
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Longitude / °E")
axes[0].set_ylabel("Latitude / °N")
fig.savefig("plots/diurnal_cycle/talk/bt_fine_coarse.png", dpi=300)

# %%
