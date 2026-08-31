# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import read_era5_vars, load_random_datasets
from scipy.stats import linregress
import numpy as np
from src.helper_functions import calculate_jj_mean, nan_detrend
from dask.diagnostics import ProgressBar
from scipy.signal import detrend

# %%
ds = read_era5_vars(mode="mean").load()
t_surf = xr.open_dataarray("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").load()
t_surf = t_surf.sel(time=ds.time)
ds_icon = load_random_datasets(version="temp")
runs = list(ds_icon.keys())
for run in runs:
    ds_icon[run] = ds_icon[run].sel(index=slice(0, 1e6))
vgrid = (
    xr.open_dataset(
        "/work/bu1562/m301049/icon-mpim/experiments/jed0001/atm_vgrid_angel.nc"
    )
    .mean("ncells")
    .rename({"height": "height_2", "height_2": "height"})
)


# %% calculate stability for icon
def calc_stability(p, t):
    R = 8.314  # J/mol/K
    cp = 29.07  # J/mol/K
    dp = p.differentiate("temp")
    dt = t.differentiate("temp")
    dt_dp = dt / dp
    stability = (t / p) * (R / cp) - dt_dp
    stability.attrs = {"long_name": "Static stability", "units": "K/Pa"}
    return stability


for run in runs:
    ds_icon[run]["stab"] = calc_stability(ds_icon[run]["pfull"], ds_icon[run]["ta"])

# %% average stability and temperature for ICON
ds_icon_average = {}
for run in runs:
    ds_icon_average[run] = ds_icon[run][["stab", "ta", "pfull"]].mean("index")

# %% regrid era5 to temperature levels
idx_trop = ds["t"].sel(hybrid=slice(10, None)).argmin("hybrid")
height_trop = ds["hybrid"].sel(hybrid=slice(10, None)).isel(hybrid=idx_trop)
mask_trop = (ds["hybrid"] > height_trop).load()

# build temperature indexer
print("Build temperature indexer")
t_grid = np.linspace(180, 260, 60)


def interpolate_height(t, hybrid):
    return np.interp(
        t_grid, t[~np.isnan(t)], hybrid[~np.isnan(t)], left=np.nan, right=np.nan
    )


# Use Dask to parallelize the interpolation
height_array = xr.apply_ufunc(
    interpolate_height,
    ds["t"].where(mask_trop),
    ds["hybrid"].where(mask_trop),
    input_core_dims=[["hybrid"], ["hybrid"]],
    output_core_dims=[["temp"]],
    vectorize=True,
    output_dtypes=[float],
    dask_gufunc_kwargs={"output_sizes": {"temp": 60}},
)

with ProgressBar():
    height_array = height_array.assign_coords(temp=t_grid, time=ds["time"]).compute()

# regrid to temperature
print("Regrid to temperature")
with ProgressBar():
    ds_regrid = ds.interp(hybrid=height_array).compute()

# %% find warmes and coldes 10% of surface temperatures
n = len(t_surf)
n_10_percent = int(0.05 * n)
sorted_indices = np.argsort(t_surf)
cold_indices = sorted_indices[:n_10_percent].values
warm_indices = sorted_indices[-n_10_percent:].values
t_surf_cold = t_surf.isel(time=cold_indices).mean("time")
t_surf_warm = t_surf.isel(time=warm_indices).mean("time")
stab_warm = ds_regrid["stability"].isel(time=warm_indices).mean("time")
stab_cold = ds_regrid["stability"].isel(time=cold_indices).mean("time")
temp_cold = ds_regrid["t"].isel(time=cold_indices).mean("time")
temp_warm = ds_regrid["t"].isel(time=warm_indices).mean("time")
t_surf_diff = t_surf_warm - t_surf_cold

# %% regression of detrended and deseasonalized stability on surface temperature
stab_detrend = xr.DataArray(
    nan_detrend(ds_regrid["stability"], "temp"),
    coords=ds_regrid["stability"].coords,
    dims=ds_regrid["stability"].dims,
)
stab_deseason = stab_detrend.groupby("time.month") - stab_detrend.groupby(
    "time.month"
).mean("time")
t_surf_detrend = xr.DataArray(detrend(t_surf), coords=t_surf.coords, dims=t_surf.dims)
t_surf_deseason = t_surf_detrend.groupby("time.month") - t_surf_detrend.groupby(
    "time.month"
).mean("time")
# %%
stab_regression = []
for temp_level in stab_deseason["temp"]:
    stab_temp = stab_deseason.sel(temp=temp_level)
    nanmask = ~np.isnan(stab_temp.values) & ~np.isnan(t_surf_deseason.values)
    slope, intercept, r_value, p_value, std_err = linregress(
        t_surf_deseason.where(nanmask).values, stab_temp.where(nanmask).values
    )
    stab_regression.append(slope)
stab_regression = xr.DataArray(
    stab_regression,
    coords={"temp": stab_deseason["temp"]},
    dims=["temp"],
    name="stab_regression",
)

# %% plot mean profiles
fig, ax = plt.subplots()
ax.plot(
    ds_icon_average["jed0011"]["stab"] * 1e5,
    ds_icon_average["jed0011"]["ta"],
    label="ICON control",
    color="#2b037c",
)
ax.plot(
    ds_icon_average["jed0022"]["stab"] * 1e5,
    ds_icon_average["jed0022"]["ta"],
    label="ICON +4K",
    color="#10f358",
)

ax.plot(
    stab_cold * 1e5,
    stab_cold["temp"],
    label="ERA5 cold",
    color="#2b037c",
    linestyle="dashed",
)
ax.plot(
    stab_warm * 1e5,
    stab_warm["temp"],
    label="ERA5 warm",
    color="#10f358",
    linestyle="dashed",
)
ax.invert_yaxis()
ax.legend()
ax.set_ylim(260, 180)
ax.set_xlim([0, 300])
# %% plot chagnes
fig, ax = plt.subplots()
ax.plot(
    stab_regression * 1e5,
    stab_regression["temp"],
    color="black",
    label="ERA5 Internal variability",
)
ax.invert_yaxis()
ax.plot(
    (stab_warm - stab_cold) * 1e5 / t_surf_diff,
    stab_cold["temp"],
    color="red",
    linestyle="dashed",
    label="ERA5 warm - cold",
)
ax.set_xlim([-5, 10])
ax.set_ylim([260, 200])
ax.plot(
    (ds_icon_average["jed0022"]["stab"] - ds_icon_average["jed0011"]["stab"]) * 1e5 / 4,
    ds_icon_average["jed0011"]["ta"],
    color="blue",
    linestyle="dashed",
    label="ICON +4K - control",
)
ax.axvline(0, color="k", linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
# %%
