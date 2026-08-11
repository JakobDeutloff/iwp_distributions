# %%
from src.helper_functions import load_slopes, load_histograms, load_cre
import xarray as xr

# %%  load data
slopes, errors, p_vals = load_slopes()
hists_obs = load_histograms("obs")
hists_model = load_histograms("model")
cre = load_cre()

# %% filter 2c_ice and dadar data for size
hists_obs["two_c_ice"] = hists_obs["two_c_ice"].where(
    hists_obs["two_c_ice"]["size"] > 1.9e6
)
hists_obs["dardar"] = hists_obs["dardar"].where(hists_obs["dardar"]["size"] > 1.9e6)

# %% calculate mean hists for obs
for key in hists_obs.keys():
    hists_obs[key] = (hists_obs[key]["hist"] / hists_obs[key]["size"]).mean("time")

# %% concat hists of obs and models
hists = {}
for key in hists_obs.keys():
    hists[key] = hists_obs[key]
hists["rcemip"] = hists_model["rcemip_control"]
hists["icon_ap"] = hists_model["icon_ap_control"]
hists["xshield"] = hists_model["xshield_control"]
hists["icon_amip"] = hists_model["icon_amip_control"]

# %% calculate feedback
feedback = xr.Dataset()
for key in slopes.keys():
    feedback = feedback.assign({key: cre["net"] * slopes[key]})

# %% partition feedback into area and opacity feedback
feedback_area = xr.Dataset()
feedback_opacity = xr.Dataset()
for key in hists.keys():
    g_cap = slopes[key].sum() / hists[key].sum()
    print(f"g_cap for {key}: {g_cap*100} %/K")
    g_prime = (slopes[key] / hists[key]) - g_cap
    feedback_area = feedback_area.assign({key: (cre["net"] * hists[key]).sum() * g_cap})
    feedback_opacity = feedback_opacity.assign({key: (g_prime * hists[key] * cre["net"]).sum()})

# %% save feedbacks
feedback.to_netcdf("/work/bu1562/m301049/iwp_dists/feedback.nc")
feedback_area.to_netcdf("/work/bu1562/m301049/iwp_dists/feedback_area.nc")
feedback_opacity.to_netcdf("/work/bu1562/m301049/iwp_dists/feedback_opacity.nc")

# %%
