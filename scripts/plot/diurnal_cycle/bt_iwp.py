# %%
import pickle
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import ccic
from tqdm import tqdm
from scipy.stats import linregress

# %%
bts = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/GPM_MERGIR/merg_200801011*.nc4", engine="netcdf4"
    )
    .sel(lat=slice(-30, 30))
    .load()
)
iwp = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/ccic/raw/ccic_cpcir_200801011*.zarr", engine="zarr"
    )
    .sel(latitude=slice(30, -30))
    .load()
)

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

# %%
# Flatten spatial dimensions before groupby
bt_flat = ds["bt"].stack(stacked=('time', 'lat', 'lon'))
iwp_flat = ds["iwp"].stack(stacked=('time', 'lat', 'lon'))
# %% plot bt vs iwp
bins = np.logspace(-3, 2, 254)[::4]
points = (bins[:-1] + bins[1:]) / 2
mean_bt = ds["bt"].groupby_bins(ds["iwp"], bins).mean()
# %%
q_05 = bt_flat.groupby_bins(iwp_flat, bins).quantile(0.05, dim='stacked')
q_95 = bt_flat.groupby_bins(iwp_flat, bins).quantile(0.95, dim='stacked')

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(points, mean_bt, color="k", label="Mean")
ax.plot(points, q_05, color="gray", linestyle="--", label="5th/95th percentile")
#ax.plot(points, q_95, color="gray", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$T_{\mathrm{b}}$ / K")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()

# %% calculate fraction of thin clouds for given Tb
bins = np.arange(197, 261, 1)
mean_lc_frac = (ds["iwp"] < 1e-1).groupby_bins(ds["bt"], bins).mean()

# %% plot
fig, ax = plt.subplots(figsize=(6, 4))
points = (bins[:-1] + bins[1:]) / 2
ax.plot(points, mean_lc_frac, color="k")
ax.set_xlabel(r"$T_{\mathrm{b}}$ / K")
ax.set_ylabel(r"Fraction of $I < 10^{-1}$ kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks([200, 230, 260])
ax.set_yticks([0, 0.3, 0.6])
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/publication/fraction_thin_clouds.pdf", bbox_inches='tight')

# %% load hists
hists = {}
hists['ccic'] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/ccic_2d_monthly_all.nc"
    )
hists['gpm'] = xr.open_dataset(
        f"/work/bm1183/m301049/diurnal_cycle_dists/gpm_2d_monthly_all.nc"
    )

# %% plot Tb(I) from histograms
area_fractions = {}
dim = {'ccic': 'iwp', 'gpm': 'bt'}
for name in ["ccic", "gpm"]:
    area_fractions[name] = hists[name]["hist"].sum(["local_time"]) / hists[name]['hist'].sum(['local_time', dim[name]])
# %%
bt_of_iwp = xr.zeros_like(area_fractions["ccic"])
bt_of_iwp['iwp'] = bt_of_iwp['iwp'][::-1]


def interp_bt_of_iwp(t):
    ccic_af = area_fractions["ccic"].sel(time=t)[::-1].cumsum("iwp").values
    gpm_af = area_fractions["gpm"].sel(time=t).cumsum("bt").values
    gpm_bt = area_fractions["gpm"].sel(time=t)["bt"].values
    return np.interp(ccic_af, gpm_af, gpm_bt)


times = area_fractions["ccic"]["time"].values
for t in tqdm(times):
    bt_of_iwp.loc[dict(time=t)] = interp_bt_of_iwp(t)

# %% make linear fit 
res = linregress(np.log10(bt_of_iwp.sel(iwp=slice(1, 1e-1))['iwp'].values), bt_of_iwp.sel(iwp=slice(1, 1e-1)).mean("time").values)

def linear_fit(iwp):
    return res.slope * iwp + res.intercept
bt_of_iwp_linear = linear_fit(np.log10(bt_of_iwp['iwp']))

# %% plot bt of iwp
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(bt_of_iwp['iwp'], bt_of_iwp.mean("time"), color="k", label="Mean")
ax.set_yticks(bt_of_iwp.mean("time").sel(iwp=[10, 1, 0.1], method="nearest").values.round(0))
ax.set_xscale("log")
ax.set_xlim(5e-2, 2e1)
ax.set_ylim(190, 270)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$T_{\mathrm{b}}$ / K")
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/publication/bt_of_iwp_area.pdf", bbox_inches='tight')

# %% save coeffs of linear fit 
with open('/work/bm1183/m301049/diurnal_cycle_dists/bt_iwp_fig_coeffs.pkl', 'wb') as f:
    pickle.dump({'slope': res.slope, 'intercept': res.intercept}, f)

# %% save bt of iwp dataset
bt_iwp_ds = xr.Dataset(
    {
        "bt_of_iwp": (("iwp", "time"), bt_of_iwp.values.T,)
    },
    coords={
        "iwp": bt_of_iwp['iwp'],
        "time": bt_of_iwp['time'],
    },
)  
bt_iwp_ds.to_netcdf('/work/bm1183/m301049/diurnal_cycle_dists/bt_of_iwp.nc')
# %%
