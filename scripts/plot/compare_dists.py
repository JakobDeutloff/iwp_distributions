# %% # %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.plot import definitions
from src.helper_functions import interpolate_bins



# %% initialize containers
bins = bins = np.logspace(-3, 2, 254)[::4]
colors, line_labels, linestyles = definitions()

# %% open distributions CCIC
path = "/work/bm1183/m301049/ccic/"
years = range(2000, 2024)
months = [f"{i:02d}" for i in range(1, 13)]
hist_list = []
for year in years:
    for month in months:
        try:
            ds = xr.open_dataset(
                f"{path}{year}/ccic_cpcir_iwp_distribution_{year}{month}.nc"
            )
            hist_list.append(ds)
        except FileNotFoundError:
            print(f"File for {year}-{month} not found, skipping.")

hists_ccic = (
    xr.concat(hist_list, dim="time").coarsen(bin_center=4, boundary="trim").sum()
)

# %% open distributions 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% open spareice
hists_spare = xr.open_dataset("/work/bm1183/m301049/spareice/hists_metop.nc")

# %% open dardar
dardar = xr.open_dataset("/work/bm1183/m301049/dardarv3.10/hist_dardar.nc")



# %% load cre data and hists from icon
cre = xr.open_dataset(
    f"/work/bm1183/m301049/icon_hcap_data/control/production/cre/jed0011_cre_raw.nc"
)

experiments = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
}
iwp_hists = {}
for run in ["jed0011", "jed0022", "jed0033"]:
    with open(
        f"/work/bm1183/m301049/icon_hcap_data/{experiments[run]}/production/{run}_iwp_hist.pkl",
        "rb",
    ) as f:
        iwp_hists[run] = pickle.load(f)
        iwp_hists[run] = xr.DataArray(
            iwp_hists[run],
            coords={"iwp": cre.iwp},
            dims=["iwp"],
        )


#  interpolate
iwp_hists_int = {}
for run in ["jed0011", "jed0022", "jed0033"]:
    iwp_hists_int[run] = interpolate_bins(iwp_hists[run], bins, name_old_bins="iwp")


# %% load rcemip data
ds = xr.open_dataset(
    "/work/bm1183/m301049/iwp_framework/blaz_adam/rcemip_iwp-resolved_statistics.nc"
)
ds["fwp"] = ds["fwp"] * 1e-3
# interpolate histogram
rcemip_pdf = interpolate_bins(
    ds["f"].mean("model").isel(SST=1), bins, name_old_bins="fwp"
)


# %%
fig, ax = plt.subplots(figsize=(10, 6))
year = "2008"

ccic = hists_ccic.sel(time=year)
spare = hists_spare.sel(time=year)
two_c = hists_2c.sel(time=year)
dardar = dardar.sel(time=year)

ax.stairs(
    ccic["hist"].sum("time") / ccic["size"].sum("time"),
    bins,
    label=line_labels["ccic"],
    color=colors["ccic"],
    linewidth=3,
    alpha=0.7,
)
ax.stairs(
    spare["hist"].sum("time") / spare["size"].sum("time"),
    bins,
    label=line_labels["spare"],
    color=colors["spare"],
    linewidth=3,
    alpha=0.7,
)
ax.stairs(
    two_c["hist"].sum("time") / two_c["size"].sum("time"),
    bins,
    label=line_labels["2c"],
    color=colors["2c"],
    linewidth=3,
    alpha=0.7,
)
ax.stairs(
    dardar["hist"].sum("time") / dardar["size"].sum("time"),
    bins,
    label=line_labels["dardar"],
    color=colors["dardar"],
    linewidth=3,
    alpha=0.7,
)
# ax.stairs(
#     rcemip_pdf,
#     bins,
#     label=line_labels["rcemip"],
#     color=colors["rcemip"],
#     linewidth=1.5,
# )
# ax.stairs(
#     iwp_hists_int["jed0011"],
#     bins,
#     label=line_labels["jed0011"],
#     color=colors["jed0011"],
#     linewidth=1.5,
# )


ax.set_xlim(1e-3, 1e2)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim([0, 0.015])
ax.set_xscale("log")
ax.set_xlim([1e-3, 20])
ax.legend()
ax.set_xlabel("$I$ / kg m$^{-2}$")
ax.set_ylabel("P(I)")
fig.tight_layout()
fig.savefig("plots/compare_dists_2008_sat.png", dpi=300)
# %%
