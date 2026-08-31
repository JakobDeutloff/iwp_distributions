# %%
import xarray as xr 
import matplotlib.pyplot as plt

# %%
icon_amip_cont = (
    xr.open_dataset(
        "/work/bu1562/m301049/icon-amip/histogram_iwp_ctrl_20200401_20200831.nc"
    )
    .rename({"iwp_bin": "iwp"})
)
icon_amip_4k = (
    xr.open_dataset(
        "/work/bu1562/m301049/icon-amip/histogram_iwp_sst4k_20200401_20200831.nc"
    )
    .rename({"iwp_bin": "iwp"})
)
# %% 
p_area_cont = icon_amip_cont['pdf'] / (icon_amip_cont['pdf'].sum(dim='iwp') + icon_amip_cont['clear_sky_area'])
p_count_cont = icon_amip_cont['counts'] / (icon_amip_cont['size'] + icon_amip_cont['clear_sky_size'])
p_area_4k = icon_amip_4k['pdf'] / (icon_amip_4k['pdf'].sum(dim='iwp') + icon_amip_4k['clear_sky_area'])
p_count_4k = icon_amip_4k['counts'] / (icon_amip_4k['size'] + icon_amip_4k['clear_sky_size'])

diff_area = p_area_4k - p_area_cont
diff_count = p_count_4k - p_count_cont

# %% plot diff area for all domains 

fig, ax = plt.subplots(figsize=(8, 6))

for domain in diff_area.domain.values:
    ax.plot(
        diff_area.iwp,
        diff_area.sel(domain=domain),
        label=domain,
    )
    ax.plot(
        diff_count.iwp,
        diff_count.sel(domain=domain),
        label=domain + " (count)",
        linestyle="--",
    )

ax.set_xscale("log")
ax.legend()


# %%
