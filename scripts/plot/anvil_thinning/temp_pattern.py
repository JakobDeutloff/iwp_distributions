# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.signal import detrend
from src.helper_functions import deseason
from scipy.stats import linregress
import numpy as np
import pandas as pd

# %% helper functions

def spatial_mean(ds):
    weights = np.cos(np.deg2rad(ds.sel(latitude=slice(30, -30)).latitude)) 
    return ds.weighted(weights).mean(["latitude", "longitude"]) 

def regression_slope(y, x):
    """Calculate regression slope"""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    x_masked = x[mask]
    y_masked = y[mask]
    return np.cov(x_masked, y_masked)[0, 1] / np.var(x_masked)

def to_annual(data):
    """Convert to annual mean with proper time coordinate."""
    annual = data.groupby("time.year").mean("time").rename({"year": "time"})
    annual["time"] = pd.to_datetime(annual.time.values, format="%Y")
    return annual


def get_linear_trend(data):
    """Calculate linear trend and return slope info and trend array."""
    annual_data = to_annual(data)
    lin_reg = linregress(
        np.arange(len(annual_data.time)), annual_data.values.flatten()
    )
    annual_trend = xr.DataArray(
        lin_reg.slope * np.arange(len(annual_data.time)) + lin_reg.intercept,
        coords=annual_data.coords,
        dims=annual_data.dims,
    )
    monthly_trend = xr.DataArray(
        np.interp(
            np.arange(len(data.time)),
            np.arange(len(annual_data.time)) * 12 + 6,
            annual_trend.values,
        ),
        coords=data.coords,
        dims=data.dims,
    )
    return monthly_trend, annual_trend, lin_reg.slope


def detrend_deseason(data, trend):
    """Detrend and deseasonalize data."""
    detrended = data - trend
    return detrended.groupby("time.month") - detrended.groupby("time.month").mean(
        "time"
    )

def calc_warming_pattern(data, mean_slope):
    """Calculate warming pattern from spatial trend."""
    trend = data.groupby('time.year').mean().polyfit(dim="year", deg=1)
    return trend["polyfit_coefficients"].sel(degree=1) / mean_slope

def calc_warming_pattern_int_var(data_2d, data_1d):
    """Calculate warming pattern from internal variability."""
    return xr.apply_ufunc(
        regression_slope,
        data_2d,
        data_1d,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

# %% load temperature data
t = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m.nc")
t = (t.sel(latitude=slice(30, -30)).rename({"valid_time": "time"}))["t2m"]
t_mean = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m
t_mean = t_mean.sel(time=t.time)

# %% load predictors
predictors = {}

max_d = xr.open_dataarray(
    "/work/bm1183/m301049/era5/monthly/max_convergence_60_95hPa.nc",
    decode_timedelta=False,
)
stability = (
    xr.open_dataarray(
        "/work/bm1183/m301049/era5/monthly/stability_at_max_convergence_60_95hPa.nc",
        decode_timedelta=False,
    )
    * 1e5
)

t_3d = xr.open_dataarray(
    "/work/bm1183/m301049/era5/monthly/t.nc",
    chunks={"longitude": 10, "hybrid": -1, "time": -1, "latitude": -1},
)
t_500 = t_3d.sel(hybrid=95).load()
t_500_mean = spatial_mean(t_500)
predictors['t'] = t_500_mean
predictors['max_d'] = spatial_mean(max_d)
predictors['stability'] = spatial_mean(stability)

# %% calculate linear regressions
t_trend, t_annual_trend, t_slope = get_linear_trend(t_mean)
t_500_trend, t_500_annual_trend, t_500_slope = get_linear_trend(t_500_mean)

# %% detrend and deseason 2D data
t_detrend_deseason = detrend_deseason(t, t_trend)
t_500_detrend_deseason = detrend_deseason(t_500, t_500_trend)

# %% detrend and deseason 1d data 
t_mean_detrend_deseason = detrend_deseason(t_mean, t_trend)
t_500_mean_detrend_deseason = detrend_deseason(t_500_mean, t_500_trend)

# %% calculate warming patterns at every grid point
warming_pattern_trend = calc_warming_pattern(t, t_slope)
warming_pattern_trend_500 = calc_warming_pattern(t_500, t_500_slope)

# %% calculate nino warming patterns
warming_pattern_nino = calc_warming_pattern_int_var(
    t_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
    t_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
)
warming_pattern_nino_500 = calc_warming_pattern_int_var(
    t_500_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
    t_500_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),  
)


# %% plot warming pattern of thrend and internal variability
fig, axes = plt.subplots(
    4,
    1,
    figsize=(10, 10),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
)
warming_pattern_trend.plot(
    ax=axes[0],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
warming_pattern_trend_500.plot(
    ax=axes[1],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
im = warming_pattern_nino.plot(
    ax=axes[2],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)
warming_pattern_nino_500.plot(
    ax=axes[3],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
)


for ax in axes:
    ax.coastlines()
    ax.set_title("")

axes[0].set_title("Linear Trend $T_{2m}$")
axes[1].set_title("Linear Trend $T_{500hPa}$")
axes[2].set_title("Internal Variability $T_{2m}$")
axes[3].set_title("Internal Variability $T_{500hPa}$")
fig.colorbar(
    im,
    ax=axes,
    orientation="horizontal",
    label="Relative Warming (K/K)",
    pad=0.05,
    aspect=50,
    extend="both",
)

# %% calculate linear regressions of other predictors
predictor_linear_trend = {}
predictor_linear_slope = {}
predictor_internal_var_slope = {}
for name, predictor in predictors.items():
    predictor_annual = to_annual(predictor)
    predictor_linear_trend[name], _, predictor_linear_slope[name] = get_linear_trend(predictor)
    predictor_detrend_deseason = detrend_deseason(predictor, predictor_linear_trend[name])
    predictor_internal_var_slope[name] = linregress(
        t_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
        predictor_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
    ).slope
    print(f"{name} linear trend: {(predictor_linear_slope[name])/t_slope:.3f}")
    print(f"{name} internal variability: {predictor_internal_var_slope[name]:.3f}")



# %% amount of FT warming per surface warming 
ft_warming_per_surface_warming_trend = t_500_slope / t_slope
ft_warming_per_surface_warming_nino = linregress(
    t_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
    t_500_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
).slope

print(f"FT warming per surface warming (trend): {ft_warming_per_surface_warming_trend:.3f}")
print(f"FT warming per surface warming (internal variability): {ft_warming_per_surface_warming_nino:.3f}")

  



# %%
