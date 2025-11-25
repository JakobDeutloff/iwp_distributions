# %% 
import xarray as xr 
import matplotlib.pyplot as plt
import ccic
import numpy as np

# %%
bts = xr.open_mfdataset('/work/bm1183/m301049/GPM_MERGIR/merg_2008010101*.nc4', engine='netcdf4').sel(lat=slice(-30,30)).load()
iwp = xr.open_mfdataset('/work/bm1183/m301049/ccic/raw/ccic_cpcir_2008010101*.zarr', engine='zarr').sel(latitude=slice(30,-30)).load()

# %% combined dataset
ds = xr.Dataset(
    {
        "bt": (("time", "lat", "lon"), bts["Tb"].values),
        "iwp": (("time", "lat", "lon"), iwp["tiwp"].values[:, ::-1, :]),
    },
    coords={
        "time": bts["time"],
        "lat": bts["lat"],
        "lon": bts["lon"],
    },
)
# %% plot bt vs iwp
bins = np.logspace(-3, 2, 254)[::4]
points = (bins[:-1] + bins[1:]) / 2
mean_bt = ds["bt"].groupby_bins(ds['iwp'], bins).mean()
std = ds["bt"].groupby_bins(ds['iwp'], bins).std()

# %%
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(points, mean_bt, color='k')
ax.fill_between(points, mean_bt - std, mean_bt + std, color='gray', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("Brightness Temperature / K")
ax.spines[['top', 'right']].set_visible(False)
fig.savefig('plots/diurnal_cycle/bt_vs_iwp.png', dpi=300)

# %%
