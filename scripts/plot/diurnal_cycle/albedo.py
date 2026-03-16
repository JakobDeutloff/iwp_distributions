# %%
import matplotlib.pyplot as plt
import xarray as xr

# %%
albedo_iwp = xr.open_dataset(
    "/work/bm1183/m301049/diurnal_cycle_dists/binned_hc_albedo_iwp.nc"
)

# %% plot albedos 
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolor(
    albedo_iwp["local_time"],
    albedo_iwp["iwp"],
    albedo_iwp["hc_albedo"],
    rasterized=True,
    vmin=0.1,
    vmax=0.8,
)
ax.set_yscale("log")
ax.set_ylim(10, 1e-2)

ax.set_xticks([6, 12, 18])
ax.set_xlabel("Local Time / h")
ax.set_ylabel("$I$ / kg m$^{-2}$")
ax.spines[["top", "right"]].set_visible(False)
cb = fig.colorbar(im, ax=ax, orientation="horizontal", label=r"$\alpha_{\mathrm{cl}}$", pad=0.15, shrink=0.8, aspect=30)
                  
cb.set_ticks([0.1, 0.45, 0.8])
fig.savefig('plots/diurnal_cycle/publication/albedo.pdf', bbox_inches='tight')
# %%
