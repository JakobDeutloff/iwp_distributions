# %%
import xarray as xr
import matplotlib.pyplot as plt
from src.helper_functions import read_era5_vars
from scipy.stats import linregress
import numpy as np
from src.helper_functions import calculate_jj_mean

# %%
ds = read_era5_vars(mode="mean").load()
t_surf = xr.open_dataarray("/work/bu1562/m301049/era5/monthly/t2m_tropics.nc").load()
t_surf = t_surf.sel(time=ds.time)

# %% calculate temporal means
ds_mean = ds.mean(dim="time")

# %% plot mean profiles
fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)

axes[0].plot(ds_mean["net_rad_tendency"], ds_mean["pressure"] / 100, color="k")
axes[0].set_xlabel("$R$ / K day$^{-1}$")
axes[0].set_xlim(-0.5, 1.7)

axes[1].plot(ds_mean["stability"] * 1000 * 100, ds_mean["pressure"] / 100, color="b")
axes[1].set_xlabel("$S$ / mK hPa$^{-1}$")
axes[1].set_xlim(0, 500)

axes[2].plot(ds_mean["subsidence"] / 100, ds_mean["pressure"] / 100, color="k")
axes[2].set_xlabel("$\omega$ / hPa day$^{-1}$")
axes[2].set_xlim(-5, 30)

axes[3].plot(ds_mean["convergence"], ds_mean["pressure"] / 100, color="red")
axes[3].set_xlabel("$D$ / day$^{-1}$")
axes[3].set_xlim(-0.1, 0.5)

axes[0].invert_yaxis()
axes[0].set_ylim(400, 80)
axes[0].set_ylabel("Pressure / hPa")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig(
    "plots/anvil_thinning/iris/stability_profiles.png", dpi=300, bbox_inches="tight"
)


# %% calculate annual means june to july
ds_jj = calculate_jj_mean(ds)
t_surf_jj = calculate_jj_mean(t_surf)

# %% cluster warmest and clodest 10 % of profiles and plot averages
t_surf_sorted = t_surf_jj.sortby(t_surf_jj)
n_profiles = t_surf_sorted.shape[0]
n_top = 3
t_surf_top = t_surf_sorted.isel(year=slice(-n_top, None))
t_surf_bottom = t_surf_sorted.isel(year=slice(0, n_top))
ds_top_raw = ds_jj.sel(year=t_surf_top.year).mean(dim="year")
ds_bottom_raw = ds_jj.sel(year=t_surf_bottom.year).mean(dim="year")

# %%  amplify differences by factor 5
diff_rad_tendency = ds_top_raw["net_rad_tendency"] - ds_bottom_raw["net_rad_tendency"]
diff_stability = ds_top_raw["stability"] - ds_bottom_raw["stability"]
diff_subsidence = ds_top_raw["subsidence"] - ds_bottom_raw["subsidence"]
diff_convergence = ds_top_raw["convergence"] - ds_bottom_raw["convergence"]

mean_rad_tendency = (
    ds_top_raw["net_rad_tendency"] + ds_bottom_raw["net_rad_tendency"]
) / 2
mean_stability = (ds_top_raw["stability"] + ds_bottom_raw["stability"]) / 2
mean_subsidence = (ds_top_raw["subsidence"] + ds_bottom_raw["subsidence"]) / 2
mean_convergence = (ds_top_raw["convergence"] + ds_bottom_raw["convergence"]) / 2

mean_temp_diff = t_surf_top.mean() - t_surf_bottom.mean()
ds_top = ds_top_raw.copy()
ds_bottom = ds_bottom_raw.copy()
factor = 2.5
ds_top["net_rad_tendency"] = mean_rad_tendency + diff_rad_tendency * factor
ds_bottom["net_rad_tendency"] = mean_rad_tendency - diff_rad_tendency * factor
ds_top["stability"] = mean_stability + diff_stability * factor
ds_bottom["stability"] = mean_stability - diff_stability * factor
ds_top["subsidence"] = mean_subsidence + diff_subsidence * factor
ds_bottom["subsidence"] = mean_subsidence - diff_subsidence * factor
ds_top["convergence"] = mean_convergence + diff_convergence * factor
ds_bottom["convergence"] = mean_convergence - diff_convergence * factor

# %% plot mean profiles for top and bottom 10 %
color_warm = "#cd000e"
color_cold = "#1f6dff"
fig, axes = plt.subplots(1, 4, figsize=(8, 4), sharey=True)
axes[0].plot(
    ds_top["net_rad_tendency"],
    ds_top["pressure"] / 100,
    color=color_warm,
)
axes[0].plot(
    ds_bottom["net_rad_tendency"],
    ds_bottom["pressure"] / 100,
    color=color_cold,
)
axes[0].set_xlabel("$Q$ / K day$^{-1}$")
axes[0].set_xlim(-0.5, 1.7)

axes[1].plot(ds_top["stability"] * 100, ds_top["pressure"] / 100, color=color_warm)
axes[1].plot(
    ds_bottom["stability"] * 100, ds_bottom["pressure"] / 100, color=color_cold
)
axes[1].set_xlabel("$S$ / K hPa$^{-1}$")
axes[1].set_xlim(0, 0.3)

axes[2].plot(ds_top["subsidence"] / 100, ds_top["pressure"] / 100, color=color_warm)
axes[2].plot(
    ds_bottom["subsidence"] / 100, ds_bottom["pressure"] / 100, color=color_cold
)
axes[2].set_xlabel("$\omega$ / hPa day$^{-1}$")
axes[2].set_xlim(-5, 30)

axes[3].plot(ds_top["convergence"], ds_top["pressure"] / 100, color=color_warm)
axes[3].plot(ds_bottom["convergence"], ds_bottom["pressure"] / 100, color=color_cold)
axes[3].set_xlabel(r"$\dfrac{\partial \omega}{\partial P}$ / day$^{-1}$")
axes[3].set_xlim(-0.1, 0.5)

axes[0].invert_yaxis()
axes[0].set_ylim(300, 100)
axes[0].set_ylabel("$P$ / hPa")
axes[0].set_yticks([300, 200, 100])
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

# add letters
for i, ax in enumerate(axes):
    ax.text(
        0.1,
        1.05,
        chr(97 + i),
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )

handles = [
    plt.Line2D([0], [0], color=color_warm),
    plt.Line2D([0], [0], color=color_cold),
]
labels = ["Warm", "Cold"]
fig.legend(
    handles,
    labels,
    bbox_to_anchor=(0.5, -0.1),
    loc="lower center",
    ncol=2,
    frameon=False,
)
fig.savefig(
    "plots/thesis/stability_profiles_top_bottom.png", dpi=300, bbox_inches="tight"
)

# %% calculate weighted mean of temperture with detrainment
t_warm = (
    ds_top_raw["t"]
    .sel(hybrid=slice(60, 90))
    .weighted(ds_top_raw["convergence"].sel(hybrid=slice(60, 90)))
    .mean(dim="hybrid")
)
t_cold = (
    ds_bottom_raw["t"]
    .sel(hybrid=slice(60, 90))
    .weighted(ds_bottom_raw["convergence"].sel(hybrid=slice(60, 90)))
    .mean(dim="hybrid")
)
print(f"Temperature at max detrainment for warm profiles: {t_warm.values:.2f} K")
print(f"Temperature at max detrainment for cold profiles: {t_cold.values:.2f} K")
print(
    f"Difference in temperature at max detrainment: {t_warm.values - t_cold.values:.2f} K"
)
print(f"Difference in surface temperature: {mean_temp_diff.values:.2f} K")
# %%
