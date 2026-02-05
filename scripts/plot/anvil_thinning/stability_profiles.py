# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import read_era5_vars
from scipy.stats import linregress
import numpy as np

# %% 
ds = read_era5_vars(mode='mean').load()
t_surf = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc").load()

# %% calculate temporal means 
ds_mean = ds.mean(dim='time')

# %% plot mean profiles 
fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)

axes[0].plot(ds_mean['net_rad_tendency'], ds_mean['pressure']/100, color='k')
axes[0].set_xlabel('Radiative Tendency / K day$^{-1}$')
axes[0].set_xlim(-0.5, 1.7)

axes[1].plot(ds_mean['stability']*1000*100, ds_mean['pressure']/100, color='k')
axes[1].set_xlabel('Stability / mK hPa$^{-1}$')
axes[1].set_xlim(0, 500)

axes[2].plot(ds_mean['subsidence']/100, ds_mean['pressure']/100, color='k')
axes[2].set_xlabel('Subsidence / hPa day$^{-1}$')
axes[2].set_xlim(-5, 30)

axes[3].plot(ds_mean['convergence'], ds_mean['pressure']/100, color='k')
axes[3].set_xlabel('Convergence / day$^{-1}$')
axes[3].set_xlim(-0.1, 0.5)

axes[0].invert_yaxis()
axes[0].set_ylim(400, 80)
axes[0].set_ylabel('Pressure / hPa')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

# %% regression of max detrainment 
max_d = ds['convergence'].sel(hybrid=slice(50, 90)).max(dim='hybrid')
max_d_level = ds['convergence'].sel(hybrid=slice(50, 90)).idxmax(dim='hybrid')
stability_at_max_d = ds['stability'].sel(hybrid=max_d_level)
t_surf = t_surf.sel(time=stability_at_max_d.time)
# %% perform linear regression
res_conv = linregress(t_surf, max_d)
x_vals_conv = np.array([t_surf.min(), t_surf.max()])
y_vals_conv = res_conv.intercept + res_conv.slope * x_vals_conv

res_stab = linregress(t_surf, stability_at_max_d*1000*100)
x_vals_stab = np.array([t_surf.min(), t_surf.max()])
y_vals_stab = res_stab.intercept + res_stab.slope * x_vals_stab

# %% scatter plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(t_surf, max_d, color='k', alpha=0.5)
axes[0].plot(x_vals_conv, y_vals_conv, color='r')
axes[0].set_ylabel('Max Convergence / day$^{-1}$')

axes[1].scatter(t_surf, stability_at_max_d*1000*100, color='k', alpha=0.5)
axes[1].plot(x_vals_stab, y_vals_stab, color='r')
axes[1].set_ylabel('Stability at Max Convergence / mK hPa$^{-1}$')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('Surface Temperature / K')



# %% calculate annual means from july to june



# %% 
