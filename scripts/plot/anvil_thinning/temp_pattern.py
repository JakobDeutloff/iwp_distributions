# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
t = xr.open_dataset("/work/bu1562/m301049/era5/monthly/t2m.nc")
t = (t.sel(latitude=slice(30, -30)).rename({"valid_time": "time"}))["t2m"]
t_mean = xr.open_dataset("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").t2m
t_mean = t_mean.sel(time=t.time)

# %% calculate linear regressions
t_trend, t_annual_trend, t_slope = get_linear_trend(t_mean)

# %% detrend and deseason 2D data
t_detrend_deseason = detrend_deseason(t, t_trend)

# %% detrend and deseason 1d data 
t_mean_detrend_deseason = detrend_deseason(t_mean, t_trend)

# %% calculate warming patterns at every grid point
warming_pattern_trend = calc_warming_pattern(t, t_slope)

# %% calculate nino warming patterns
warming_pattern_nino = calc_warming_pattern_int_var(
    t_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
    t_mean_detrend_deseason.sel(time=slice("2000-02-01", "2023-12-01")),
)

# %% plot warming pattern of thrend and internal variability
fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
)
warming_pattern_trend.plot(
    ax=axes[0],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    rasterized=True,
)
im = warming_pattern_nino.plot(
    ax=axes[1],
    cmap="bwr",
    vmin=0,
    vmax=2,
    transform=ccrs.PlateCarree(),
    add_colorbar=False,
    rasterized=True,
)


for ax in axes:
    ax.coastlines()
    ax.set_title("")

axes[0].set_title("Linear Trend")
axes[1].set_title("Internal Variability")
fig.colorbar(
    im,
    ax=axes,
    orientation="horizontal",
    label="Relative Warming  / K K$^{-1}$",
    pad=0.05,
    aspect=50,
    extend="both",
)

# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.05,
        1.2,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )

fig.savefig("plots/anvil_thinning/publication/temp_pattern.pdf", bbox_inches="tight", dpi=300)

# %%
