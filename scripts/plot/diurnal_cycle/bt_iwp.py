# %% 
import xarray as xr 
import matplotlib.pyplot as plt
import ccic
import numpy as np
from scipy.stats import linregress
import pickle

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

# %% fit straight line between iwp = 10 and iwp = 0.1
res = linregress(
    np.log10(points[(points >= 0.1) & (points <= 10)]),
    mean_bt[(points >= 0.1) & (points <= 10)],
)
x = np.log10(points[(points >= 0.1) & (points <= 10)])
y_fit = res.slope * x + res.intercept
x_pot = 10**x

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(points, mean_bt, color='k', label='Mean')
ax.fill_between(points, mean_bt - std, mean_bt + std, color='gray', alpha=0.5, label='± $\sigma$')
ax.plot(x_pot, y_fit, color='red', linestyle='--', label='Linear fit')
ax.set_xscale('log')
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$T_{\mathrm{b}}$ / K")
ax.spines[['top', 'right']].set_visible(False)
ax.legend()
fig.savefig('plots/diurnal_cycle/bt_vs_iwp.png', dpi=300)

# %% save coeffs of linear fit
with open('/work/bm1183/m301049/diurnal_cycle_dists/bt_iwp_fig_coeffs.pkl', 'wb') as f:
    pickle.dump({'slope': res.slope, 'intercept': res.intercept}, f)

# %%
