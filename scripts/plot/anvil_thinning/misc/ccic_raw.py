# %%
import ccic
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import s3fs

# %% load dardar
dardar_raw = xr.open_dataset("/work/bm1183/m301049/dardar/dardar_iwp_2008.nc")
mask = (dardar_raw["latitude"] > -20) & (dardar_raw["latitude"] < 20)
dardar_raw = dardar_raw.where(mask)

# %% load ccic
ds = xr.open_mfdataset("/work/bm1183/m301049/ccic/*.zarr", engine="zarr")
ds = ds["tiwp"].sel(latitude=slice(20, -20)).load()


# %% get histograms
hists = {}
bins = np.logspace(-3, 2, 254)[::4]
hists["dardar"], edges = np.histogram(
    dardar_raw["iwp"] * 1e-3, bins=bins, density=False
)
hists["dardar"] = (
    hists["dardar"] / dardar_raw["iwp"].count().values
)  # Normalize histogram
hists["ccic"], edges = np.histogram(
    ds.values, bins=bins, density=False
)
hists["ccic"] = hists["ccic"] / ds.size
# %%
fig, ax = plt.subplots(figsize=(10, 5))

ax.stairs(hists["ccic"], edges, color="red", label="CCIC CPCIR")
ax.stairs(hists["dardar"], edges, color="blue", label="DARDAR v2.1")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$P(I)$")
ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.savefig("plots/ccic_histogram.png", bbox_inches="tight", dpi=300)
# %%
