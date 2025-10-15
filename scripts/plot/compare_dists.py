# %% # %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


# %% initialize containers
bins = bins = np.logspace(-3, 2, 254)[::4]

# %% open distributions CCIC
path = "/work/bm1183/m301049/ccic/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hists_ccic = xr.concat(hist_list, dim="time").coarsen(bin_center=4, boundary="trim").sum()

# %% open distributions 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% open spareice 
hists_spare = xr.open_dataset('/work/bm1183/m301049/spareice/hists_metop.nc')

# %% open dardar
dardar_raw = xr.open_dataset("/work/bm1183/m301049/dardar/dardar_iwp_2008.nc")
mask = (dardar_raw["latitude"] > -30) & (dardar_raw["latitude"] < 30)
dardar_raw = dardar_raw.where(mask)

# %% calculate dardar hist
dardar, edges = np.histogram(
    dardar_raw["iwp"] * 1e-3, bins=bins, density=False
)
dardar = (
    dardar / dardar_raw["iwp"].count().values
)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
year = '2008'

ccic = hists_ccic.sel(time=year)
spare = hists_spare.sel(time=year)
two_c = hists_2c.sel(time=year)

ax.stairs(ccic['hist'].sum('time') / ccic['size'].sum('time'), bins, label='CCIC', color='blue')
ax.stairs(spare['hist'].sum('time') / spare['sizes'].sum('time'), bins, label='SPARE-ICE', color='red')
ax.stairs(two_c['hist'].sum('time') / two_c['size'].sum('time'), bins, label='2C-ICE', linewidth=3, color='grey')
ax.stairs(dardar, edges, label='DARDAR v2.1', linewidth=3, color='brown')

ax.set_xlim(1e-3, 1e2)
ax.spines[['top', 'right']].set_visible(False)

ax.set_xscale('log')
ax.legend()
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("P(I)")
fig.savefig("plots/compare_dists_2008.png", dpi=300)
# %%
