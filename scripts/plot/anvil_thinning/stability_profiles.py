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
fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)

axes[0].plot(ds_mean['net_rad_tendency'], ds_mean['pressure']/100, color='k')
axes[0].set_xlabel('$R$ / K day$^{-1}$')
axes[0].set_xlim(-0.5, 1.7)

axes[1].plot(ds_mean['stability']*1000*100, ds_mean['pressure']/100, color='b')
axes[1].set_xlabel('$S$ / mK hPa$^{-1}$')
axes[1].set_xlim(0, 500)

axes[2].plot(ds_mean['subsidence']/100, ds_mean['pressure']/100, color='k')
axes[2].set_xlabel('$\omega$ / hPa day$^{-1}$')
axes[2].set_xlim(-5, 30)

axes[3].plot(ds_mean['convergence'], ds_mean['pressure']/100, color='red')
axes[3].set_xlabel('$D$ / day$^{-1}$')
axes[3].set_xlim(-0.1, 0.5)

axes[0].invert_yaxis()
axes[0].set_ylim(400, 80)
axes[0].set_ylabel('Pressure / hPa')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

fig.savefig('plots/anvil_thinning/iris/stability_profiles.png', dpi=300, bbox_inches='tight')


# %%
