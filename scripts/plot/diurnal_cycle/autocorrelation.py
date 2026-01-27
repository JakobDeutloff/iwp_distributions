# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import deseason
from scipy.signal import detrend
from scipy.stats import pearsonr
import numpy as np

# %% load era5 surface temp
temp = xr.open_dataset("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").t2m

# %%  detrend and deseasonalize
temp_detrend = xr.DataArray(detrend(temp), coords=temp.coords, dims=temp.dims)
temp_detrend = deseason(temp_detrend)

# %% calculate autocorrelation for different lags
lags = np.arange(1, 6 * 12 + 1)
autocorrs = []
for lag in lags:
    temp_lagged = temp_detrend.shift(time=lag)
    valid = ~np.isnan(temp_detrend) & ~np.isnan(temp_lagged)
    corr = pearsonr(
        temp_detrend.where(valid, drop=True).values.flatten(),
        temp_lagged.where(valid, drop=True).values.flatten(),
    )[0]
    autocorrs.append(corr)

# %% plot autocorrelation
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(temp_detrend.time, temp_detrend, color="k")
axes[1].axhline(0, color="k", linestyle="-", linewidth=0.5)
axes[1].plot(lags / 12, autocorrs, color="k")
axes[1].set_ylabel("Autocorrelation")
axes[1].set_xlabel("Lag / Years")
axes[0].set_xlabel("Time")
axes[0].set_ylabel(r"$T$ / K")
for ax, letter in zip(axes, ["a", "b"]):
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.08,
        0.9,
        letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
    )
fig.savefig('plots/diurnal_cycle/publication/t2m_autocorrelation.pdf', bbox_inches='tight')

# %%
