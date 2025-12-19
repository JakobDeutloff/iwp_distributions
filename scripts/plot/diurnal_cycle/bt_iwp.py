# %%
import xarray as xr
import matplotlib.pyplot as plt
import ccic
import numpy as np
from scipy.stats import linregress
import pickle
import glob
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# %%
bts = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/GPM_MERGIR/merg_2008010101*.nc4", engine="netcdf4"
    )
    .sel(lat=slice(-30, 30))
    .load()
)
iwp = (
    xr.open_mfdataset(
        "/work/bm1183/m301049/ccic/raw/ccic_cpcir_2008010101*.zarr", engine="zarr"
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
# %% plot bt vs iwp
bins = np.logspace(-3, 2, 254)[::4]
points = (bins[:-1] + bins[1:]) / 2
mean_bt = ds["bt"].groupby_bins(ds["iwp"], bins).mean()
std = ds["bt"].groupby_bins(ds["iwp"], bins).std()

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
ax.plot(points, mean_bt, color="k", label="Mean")
ax.fill_between(
    points, mean_bt - std, mean_bt + std, color="gray", alpha=0.5, label="± $\sigma$"
)
ax.plot(x_pot, y_fit, color="red", linestyle="--", label="Linear fit")
ax.set_xscale("log")
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$T_{\mathrm{b}}$ / K")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/bt_vs_iwp.png", dpi=300)

# %% save coeffs of linear fit
with open("/work/bm1183/m301049/diurnal_cycle_dists/bt_iwp_fig_coeffs.pkl", "wb") as f:
    pickle.dump({"slope": res.slope, "intercept": res.intercept}, f)

# %% load hists
hists = {}
files = glob.glob(
    "/work/bm1183/m301049/ccic_daily_cycle/*/ccic_cpcir_daily_cycle_distribution_2d*.nc"
)
files_all = [f for f in files if re.search(r"2d_\d{4}\.nc$", f)]
hists["ccic"] = xr.open_mfdataset(files_all).load()
hists["gpm"] = xr.open_mfdataset(
    "/work/bm1183/m301049/GPM_MERGIR/hists/gpm_*.nc"
).load()

# %% plot Tb(I) from histograms
area_fractions = {}
for name in ["ccic", "gpm"]:
    area_fractions[name] = hists[name]["hist"].sum(["local_time"]) / hists[name]["size"]
# %%
n = 3000
bt_of_iwp = xr.zeros_like(area_fractions["ccic"].isel(time=slice(None, n)))


def interp_bt_of_iwp(t):
    ccic_af = area_fractions["ccic"].sel(time=t)[::-1].cumsum("iwp").values
    gpm_af = area_fractions["gpm"].sel(time=t).cumsum("bt").values
    gpm_bt = area_fractions["gpm"].sel(time=t)["bt"].values
    return np.interp(ccic_af, gpm_af, gpm_bt)


times = area_fractions["ccic"]["time"].values[:n]
for t in tqdm(times):
    bt_of_iwp.loc[dict(time=t)] = interp_bt_of_iwp(t)

# %% plot bt of iwp
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(hists["ccic"]["iwp"][::-1], bt_of_iwp.mean("time"), color="k")
ax.fill_between(
    hists["ccic"]["iwp"][::-1],
    bt_of_iwp.mean("time") - bt_of_iwp.std("time"),
    bt_of_iwp.mean("time") + bt_of_iwp.std("time"),
    color="gray",
    alpha=0.5,
)
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e1)
ax.set_ylim(199, 290)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("$T_{\mathrm{b}}$ / K")
fig.tight_layout()
fig.savefig("plots/diurnal_cycle/bt_of_iwp_area.png", dpi=300)


# %%
