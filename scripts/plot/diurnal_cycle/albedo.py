# %%
import matplotlib.pyplot as plt
import xarray as xr

# %%
albedo_iwp = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc"
)
albedo_bt = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_bt.nc"
)

# %% plot albedos 
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im = axes[0].pcolor(
    albedo_iwp["local_time"],
    albedo_iwp["iwp"],
    albedo_iwp["hc_albedo"],
    rasterized=True,
    vmin=0.2,
    vmax=0.8,
)
axes[0].set_yscale("log")
axes[0].set_ylim(10, 1e-1)


axes[1].pcolor(
    albedo_bt["local_time"],
    albedo_bt["bt"],
    albedo_bt["hc_albedo"],
    rasterized=True,
    vmin=0.2,
    vmax=0.8,
)
axes[1].set_ylim(200, 260)
axes[1].set_yticks([200, 230, 260])
axes[0].set_ylabel("$I$ / kg m$^{-2}$")
axes[1].set_ylabel(r"$T_{\mathrm{b}}$ / K")
for ax in axes:
    ax.set_xticks([6, 12, 18])
    ax.set_xlabel("Local Time / h")
    ax.spines[["top", "right"]].set_visible(False)
cb = fig.colorbar(im, ax=axes, orientation="horizontal", label="High Cloud Albedo", pad=0.15, shrink=0.8, aspect=30)
                  
cb.set_ticks([0.2, 0.4, 0.6, 0.8])
fig.savefig('plots/diurnal_cycle/publication/albedo.pdf', bbox_inches='tight')
# %%
