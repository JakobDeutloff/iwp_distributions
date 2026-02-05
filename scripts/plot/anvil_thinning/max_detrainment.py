# %% 
import xarray as xr 
import matplotlib.pyplot as plt
from src.helper_functions import calculate_jj_mean
from scipy.stats import linregress
import numpy as np

# %%
max_d = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/max_convergence_50_90hPa.nc", decode_timedelta=False)
stability_at_max_d = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/stability_at_max_convergence_50_90hPa.nc", decode_timedelta=False) * 1e5
max_d_level = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/level_of_max_convergence_50_90hPa.nc", decode_timedelta=False)
t_surf = xr.open_dataarray("/work/bm1183/m301049/era5/monthly/t2m_tropics.nc", decode_timedelta=False)
t_surf = t_surf.sel(time=max_d.time)


# %% calculate tropical means 
max_d_mean = max_d.mean(dim=['latitude', 'longitude'])
stability_at_max_d_mean = stability_at_max_d.mean(dim=['latitude', 'longitude'])

# %% perform linear regression
res_conv = linregress(stability_at_max_d_mean, max_d_mean)
x_vals_conv = np.array([stability_at_max_d_mean.min(), stability_at_max_d_mean.max()])
y_vals_conv = res_conv.intercept + res_conv.slope * x_vals_conv
res_stab = linregress(t_surf, stability_at_max_d_mean)
x_vals_stab = np.array([t_surf.min(), t_surf.max()])
y_vals_stab = res_stab.intercept + res_stab.slope * x_vals_stab
res_conv_temp = linregress(t_surf, max_d_mean)
x_vals_conv_temp = np.array([t_surf.min(), t_surf.max()])
y_vals_conv_temp = res_conv_temp.intercept + res_conv_temp.slope * x_vals_conv_temp

# %% scatterplot 
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].scatter(t_surf, stability_at_max_d_mean, color='k', alpha=0.5)
axes[0].plot(x_vals_stab, y_vals_stab, color='r')
axes[0].text(0.05, 0.95, f"R={res_stab.rvalue:.2f}", transform=axes[0].transAxes, verticalalignment='top')
axes[0].set_ylabel('Stability at Max Convergence / mK hPa$^{-1}$')
axes[0].set_xlabel('Surface Temperature / K')

axes[1].scatter(t_surf, max_d_mean, color='k', alpha=0.5)
axes[1].plot(x_vals_conv_temp, y_vals_conv_temp, color='r')
axes[1].text(0.05, 0.95, f"R={res_conv_temp.rvalue:.2f}", transform=axes[1].transAxes, verticalalignment='top')
axes[1].set_ylabel('Max Convergence / day$^{-1}$')
axes[1].set_xlabel('Surface Temperature / K')

axes[2].scatter(stability_at_max_d_mean, max_d_mean, color='k', alpha=0.5)
axes[2].plot(x_vals_conv, y_vals_conv, color='r')
axes[2].text(0.05, 0.95, f"R={res_conv.rvalue:.2f}", transform=axes[2].transAxes, verticalalignment='top')
axes[2].set_ylabel('Max Convergence / day$^{-1}$')
axes[2].set_xlabel('Stability at Max Convergence / mK hPa$^{-1}$')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)


# %% calculate jj means 
max_d_jj = calculate_jj_mean(max_d_mean)
stability_at_max_d_jj = calculate_jj_mean(stability_at_max_d_mean)
t_surf_jj = calculate_jj_mean(t_surf)

# %% calculate linear regression for jj means
res_conv_jj = linregress(stability_at_max_d_jj, max_d_jj)
x_vals_conv_jj = np.array([stability_at_max_d_jj.min(), stability_at_max_d_jj.max()])
y_vals_conv_jj = res_conv_jj.intercept + res_conv_jj.slope * x_vals_conv_jj
res_stab_jj = linregress(t_surf_jj, stability_at_max_d_jj)
x_vals_stab_jj = np.array([t_surf_jj.min(), t_surf_jj.max()])
y_vals_stab_jj = res_stab_jj.intercept + res_stab_jj.slope * x_vals_stab_jj
res_conv_temp_jj = linregress(t_surf_jj, max_d_jj)
x_vals_conv_temp_jj = np.array([t_surf_jj.min(), t_surf_jj.max()])
y_vals_conv_temp_jj = res_conv_temp_jj.intercept + res_conv_temp_jj.slope * x_vals_conv_temp_jj

# %% scatterpolt of jj means 
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

axes[0].scatter(t_surf_jj, stability_at_max_d_jj, color='k', alpha=0.5)
axes[0].plot(x_vals_stab_jj, y_vals_stab_jj, color='r')
axes[0].text(0.05, 0.95, f"R={res_stab_jj.rvalue:.2f}", transform=axes[0].transAxes, verticalalignment='top')
axes[0].set_ylabel('Stability at Max Convergence / mK hPa$^{-1}$')
axes[0].set_xlabel('Surface Temperature / K')

axes[1].scatter(t_surf_jj, max_d_jj, color='k', alpha=0.5)
axes[1].plot(x_vals_conv_temp_jj, y_vals_conv_temp_jj, color='r')
axes[1].text(0.05, 0.95, f"R={res_conv_temp_jj.rvalue:.2f}", transform=axes[1].transAxes, verticalalignment='top')
axes[1].set_ylabel('Max Convergence / day$^{-1}$')
axes[1].set_xlabel('Surface Temperature / K')

axes[2].scatter(stability_at_max_d_jj, max_d_jj, color='k', alpha=0.5)
axes[2].plot(x_vals_conv_jj, y_vals_conv_jj, color='r')
axes[2].text(0.05, 0.95, f"R={res_conv_jj.rvalue:.2f}", transform=axes[2].transAxes, verticalalignment='top')
axes[2].set_ylabel('Max Convergence / day$^{-1}$')
axes[2].set_xlabel('Stability at Max Convergence / mK hPa$^{-1}$')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

# %%
