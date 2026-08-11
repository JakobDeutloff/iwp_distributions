# %%
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from src.helper_functions import nan_detrend, deseason, load_histograms
from src.plot import definitions
from scipy.signal import detrend
from scipy.stats import linregress

# %% load predictors
colors, line_labels, linestyles = definitions()
predictors = {}
predictors['max_d'] = xr.open_dataarray(
    "/work/bm1183/m301049/era5/monthly/max_convergence_60_95hPa.nc",
    decode_timedelta=False,
).mean(["latitude", "longitude"])
predictors['stability'] = (
    xr.open_dataarray(
        "/work/bm1183/m301049/era5/monthly/stability_at_max_convergence_60_95hPa.nc",
        decode_timedelta=False,
    ).mean(["latitude", "longitude"])
    * 1e5
)
predictors['t_surf'] = xr.open_dataarray(
    "/work/bm1183/m301049/era5/monthly/t2m_tropics.nc", decode_timedelta=False
)
predictors['t_surf'] = predictors['t_surf'].sel(time=predictors['max_d'].time)

# %% load hists
hists = load_histograms()

# %% filter 2c_ice and dadar data for size
hists['two_c_ice'] = hists['two_c_ice'].where(hists['two_c_ice']["size"] > 1.9e6)
hists['dardar'] = hists['dardar'].where(hists['dardar']["size"] > 1.9e6)

# %% normalise hists
hists_normalized = {}
for key in hists.keys():
    hists_normalized[key] = hists[key]["hist"] / hists[key]["size"]

# %%  detrend and deseasonalize
predictors_detrend = {}
for predictor_name, predictor in predictors.items():
    predictor_detrend = xr.DataArray(
        detrend(predictor), coords=predictor.coords, dims=predictor.dims
    )
    predictor_detrend = deseason(predictor_detrend)
    predictors_detrend[predictor_name] = predictor_detrend

hists_deseason = {}

for key in hists_normalized.keys():
    hists_detrend = nan_detrend(hists_normalized[key])
    hists_deseason_ds = hists_detrend.groupby("time.month") - hists_detrend.groupby(
        "time.month"
    ).mean("time")
    hists_deseason_ds["time"] = pd.to_datetime(
        hists_deseason_ds["time"].dt.strftime("%Y-%m")
    )
    hists_deseason[key] = hists_deseason_ds

# %% # %%regression
slopes_all = {}
err_all = {}
r_all = {}

for key in hists_deseason.keys():
    slopes = {}
    err = {}
    r = {}
    for predictor_name, predictor in predictors_detrend.items():
        slopes_ds = []
        err_ds = []
        r_ds = []
        hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
        predictor = predictor.sel(time=hist_vals.time)
        for i in range(hists_deseason[key].iwp.size):
            hist_row = hist_vals.isel(iwp=i).values
            res = linregress(predictor.values, hist_row)
            slopes_ds.append(res.slope)
            err_ds.append(res.stderr)
            r_ds.append(res.rvalue)
        slopes[predictor_name] = xr.DataArray(
            slopes_ds,
            coords={"iwp": hists_deseason[key].iwp},
            dims=["iwp"],
        )
        err[predictor_name] = xr.DataArray(
            err_ds,
            coords={"iwp": hists_deseason[key].iwp},
            dims=["iwp"],
        )
        r[predictor_name] = xr.DataArray(
            r_ds,
            coords={"iwp": hists_deseason[key].iwp},
            dims=["iwp"],
        )
    r['max_d'] = r['max_d'] * -1
    slopes_all[key] = slopes
    err_all[key] = err
    r_all[key] = r

# %% remove influence of D and regress residuals on T
residuals = {}
for key in hists_deseason.keys():
    hist_vals = hists_deseason[key].where(hists_deseason[key].notnull(), drop=True)
    predictor_d = predictors_detrend["max_d"].sel(time=hist_vals.time)
    residuals_ds = []
    for i in range(hists_deseason[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(predictor_d.values, hist_row)
        residuals_ds.append(hist_row - (res.slope * predictor_d.values + res.intercept))
    residuals[key] = xr.DataArray(
        residuals_ds,
        coords={"iwp": hists_deseason[key].iwp, "time": hist_vals.time},
        dims=["iwp", "time"],
    )

slopes_residuals = {}
r_residuals = {}
for key in hists_deseason.keys():
    hist_vals = residuals[key].where(residuals[key].notnull(), drop=True)
    predictor_t = predictors_detrend["t_surf"].sel(time=hist_vals.time)
    slopes_ds = []
    r_ds = []
    for i in range(residuals[key].iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(predictor_t.values, hist_row)
        slopes_ds.append(res.slope)
        r_ds.append(res.rvalue)

    slopes_residuals[key] = xr.DataArray(
        slopes_ds,
        coords={"iwp": residuals[key].iwp},
        dims=["iwp"],
    )
    r_residuals[key] = xr.DataArray(
        r_ds,
        coords={"iwp": residuals[key].iwp},
        dims=["iwp"],
    )


# %% plot only r values for annual T and D 
fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharex=True, sharey=True)

for i, key in enumerate(hists.keys()):
    axes[i].plot(
        r_all[key]['t_surf'].sel(iwp=slice(1e-3, 20)).iwp,
        r_all[key]['t_surf'].sel(iwp=slice(1e-3, 20)),
        label='$\mathrm{d}f(I) / \mathrm{d}T$',
        color='k',
    )
    axes[i].plot(
        r_all[key]['max_d'].sel(iwp=slice(1e-3, 20)).iwp,
        r_all[key]['max_d'].sel(iwp=slice(1e-3, 20)),
        label='$\mathrm{d}f(I) / \mathrm{d}D$',
        color='r',
    )
    axes[i].plot(
        r_residuals[key].sel(iwp=slice(1e-3, 20)).iwp,
        r_residuals[key].sel(iwp=slice(1e-3, 20)),
        label='$\mathrm{d}f(I) / \mathrm{d}T$ (D removed)',
        color='grey',
    )
    axes[i].set_xscale("log")
    axes[i].set_xlabel('$I$ / kg m$^{-2}$')
    axes[i].axhline(0, color='k', linewidth=0.5)
    axes[i].spines[["top", "right"]].set_visible(False)
    axes[i].set_title(line_labels[key])


axes[0].set_ylabel("r-value")
axes[0].set_yticks([-0.5, 0, 0.5])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.7, -0.1), ncols=3, frameon=False)
fig.savefig('plots/anvil_thinning/r_values_monthly.pdf', bbox_inches='tight')

# %% calculate slopes with respect to temperature for monthly values
max_d_temp = linregress(predictors_detrend["t_surf"].values, predictors_detrend["max_d"].values)
stab_temp = linregress(predictors_detrend["t_surf"].values, predictors_detrend["stability"].values)
max_d_stab = linregress(predictors_detrend["stability"].values, predictors_detrend["max_d"].values)

slopes_predictors = {
    "max_d": max_d_temp.slope,
    "stability": stab_temp.slope,
    "max_d_stab": max_d_stab.slope,
}
intercepts_predictors = {
    "max_d": max_d_temp.intercept,
    "stability": stab_temp.intercept,
    "max_d_stab": max_d_stab.intercept,
}

slopes_temp = {}
err_temp = {}
for predictor_name in ["max_d", "stability"]:
    slopes_temp[predictor_name] = (
        slopes[predictor_name] * slopes_predictors[predictor_name]
    )
    err_temp[predictor_name] = err[predictor_name] * slopes_predictors[predictor_name]

slopes_temp["max_d_stab"] = (
    slopes["max_d"] * slopes_predictors["max_d_stab"] * slopes_predictors["stability"]
)
err_temp["max_d_stab"] = (
    err["max_d"] * slopes_predictors["max_d_stab"] * slopes_predictors["stability"]
)


# %% plot slopes in one plot 
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(0, color='k', linewidth=0.5)
ax.plot(slopes['t_surf'].sel(iwp=slice(1e-3, None)).iwp, slopes['t_surf'].sel(iwp=slice(1e-3, None)), label='$\partial f / \partial T$', color='k')
ax.plot(slopes_temp['max_d'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp['max_d'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial D \cdot \partial D / \partial T$', color='r')
ax.plot(slopes_temp['stability'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp['stability'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial S \cdot \partial S / \partial T$', color='b')
ax.plot(slopes_temp['max_d_stab'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp['max_d_stab'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial D \cdot \partial D / \partial S \cdot \partial S / \partial T$', color='g')
ax.set_xscale("log")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
ax.set_xlabel('$I$ / kg m$^{-2}$')
ax.set_ylabel('$\partial f / \partial T$ / K$^{-1}$')
ax.set_ylim(-0.0008, 0.0002)
ax.set_title('Monthly Anomalies')
fig.savefig('plots/anvil_thinning/iris/slopes_monthly.png', dpi=300, bbox_inches='tight')

# %% make scatterplot of predictors and slopes
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(predictors_detrend["t_surf"].values, predictors_detrend["stability"].values, color='b', alpha=0.5)
axes[0].set_xlabel("T / K")
axes[0].set_ylabel("S / mK hPa$^{-1}$")
axes[1].scatter(predictors_detrend["t_surf"].values, predictors_detrend["max_d"].values, color='r', alpha=0.5)
axes[1].set_xlabel("T / K")
axes[1].set_ylabel("D / day$^{-1}$")
axes[2].scatter(predictors_detrend["stability"].values, predictors_detrend["max_d"].values, color='g', alpha=0.5)
axes[2].set_xlabel("S / mK hPa$^{-1}$")
axes[2].set_ylabel("D / day$^{-1}$") 

axes[1].plot(predictors_detrend["t_surf"].values, predictors_detrend["t_surf"].values * slopes_predictors["max_d"] + intercepts_predictors["max_d"], color='k')
axes[0].plot(predictors_detrend["t_surf"].values, predictors_detrend["t_surf"].values * slopes_predictors["stability"] + intercepts_predictors["stability"], color='k')
axes[2].plot(predictors_detrend["stability"].values, predictors_detrend["stability"].values * slopes_predictors["max_d_stab"] + intercepts_predictors["max_d_stab"], color='k') 

axes[0].text(0.05, 0.95, f"$\partial S / \partial T = ${slopes_predictors['stability']:.2e} mK hPa$^{{-1}}$ K$^{{-1}}$", transform=axes[0].transAxes, verticalalignment='top')
axes[1].text(0.05, 0.95, f"$\partial D / \partial T = ${slopes_predictors['max_d']:.2e} day$^{{-1}}$ K$^{{-1}}$", transform=axes[1].transAxes, verticalalignment='top')
axes[2].text(0.05, 0.95, f"$\partial D / \partial S = ${slopes_predictors['max_d_stab']:.2e} hPa day$^{{-1}}$ mK$^{{-1}}$", transform=axes[2].transAxes, verticalalignment='top')
fig.tight_layout()
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle('Monthly Anomalies')

fig.savefig('plots/anvil_thinning/iris/predictor_scatter_monthly.png', dpi=300, bbox_inches='tight')

# %% calculat regression from annual means 
hist_annual = hist.groupby("time.year").sum("time") #calculate_jj_sum(hist)
hist_annual_norm = hist_annual["hist"] / hist_annual["size"]
hist_annual_detrend = nan_detrend(hist_annual_norm, dim="iwp")

predictors_annual = {}
for predictor_name, predictor in predictors.items():
    predictor_annual = predictor.groupby("time.year").mean("time") #calculate_jj_mean(predictor)
    predictor_annual_detrend = xr.DataArray(
        detrend(predictor_annual), coords=predictor_annual.coords, dims=predictor_annual.dims
    )
    predictors_annual[predictor_name] = predictor_annual_detrend

slopes_annual = {}
err_annual = {}
r_annual = {}
for predictor_name, predictor in predictors_annual.items():
    slopes_ds = []
    err_ds = []
    r_ds = []
    hist_vals = hist_annual_detrend.where(hist_annual_detrend.notnull(), drop=True)
    predictor = predictor.sel(year=hist_vals.year)
    for i in range(hist_annual_detrend.iwp.size):
        hist_row = hist_vals.isel(iwp=i).values
        res = linregress(predictor.values, hist_row)
        slopes_ds.append(res.slope)
        err_ds.append(res.stderr)
        r_ds.append(res.rvalue)
    slopes_annual[predictor_name] = xr.DataArray(
        slopes_ds,
        coords={"iwp": hist_annual_detrend.iwp},
        dims=["iwp"],
    )
    err_annual[predictor_name] = xr.DataArray(
        err_ds,
        coords={"iwp": hist_annual_detrend.iwp},
        dims=["iwp"],
    )
    r_annual[predictor_name] = xr.DataArray(
        r_ds,
        coords={"iwp": hist_annual_detrend.iwp},
        dims=["iwp"],
    )
r_annual['max_d'] = r_annual['max_d'] * -1

# %% calculate slopes of annual predictors 
slopes_predictors_annual = {
    "max_d": linregress(predictors_annual["t_surf"].values, predictors_annual["max_d"].values).slope,
    "stability": linregress(predictors_annual["t_surf"].values, predictors_annual["stability"].values).slope,
    "max_d_stab": linregress(predictors_annual["stability"].values, predictors_annual["max_d"].values).slope,
}
intercepts_predictors_annual = {
    "max_d": linregress(predictors_annual["t_surf"].values, predictors_annual["max_d"].values).intercept,
    "stability": linregress(predictors_annual["t_surf"].values, predictors_annual["stability"].values).intercept,
    "max_d_stab": linregress(predictors_annual["stability"].values, predictors_annual["max_d"].values).intercept,
}

slopes_temp_annual = {}
err_temp_annual = {}
for predictor_name in ["max_d", "stability"]:
    slopes_temp_annual[predictor_name] = (
        slopes_annual[predictor_name] * slopes_predictors_annual[predictor_name]
    )
    err_temp_annual[predictor_name] = err_annual[predictor_name] * slopes_predictors_annual[predictor_name]
slopes_temp_annual["max_d_stab"] = (
    slopes_annual["max_d"] * slopes_predictors_annual["max_d_stab"] * slopes_predictors_annual["stability"]
)
err_temp_annual["max_d_stab"] = (
    err_annual["max_d"] * slopes_predictors_annual["max_d_stab"] * slopes_predictors_annual["stability"]
)

# %% plot annual slopes in one plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(0, color='k', linewidth=0.5)
ax.plot(slopes_annual['t_surf'].sel(iwp=slice(1e-3, None)).iwp, slopes_annual['t_surf'].sel(iwp=slice(1e-3, None)), label='$\partial f / \partial T$', color='k')
ax.plot(slopes_temp_annual['max_d'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp_annual['max_d'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial D \cdot \partial D / \partial T$', color='r')
ax.plot(slopes_temp_annual['stability'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp_annual['stability'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial S \cdot \partial S / \partial T$', color='b')
ax.plot(slopes_temp_annual['max_d_stab'].sel(iwp=slice(1e-3, None)).iwp, slopes_temp_annual['max_d_stab'].sel(iwp=slice(1e-3, None)), label='$\partial f/\partial D \cdot \partial D / \partial S \cdot \partial S / \partial T$', color='g')
ax.set_xscale("log")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(-0.0008, 0.0002)
ax.set_xlabel('$I$ / kg m$^{-2}$')
ax.set_ylabel('$\partial f / \partial T$ / K$^{-1}$')
ax.set_title('Annual Anomalies')
fig.savefig('plots/anvil_thinning/iris/slopes_annual.png', dpi=300, bbox_inches='tight')

# %% plot scatter of predictors and slopes for annual means
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(predictors_annual["t_surf"].values, predictors_annual["stability"].values, color='b', alpha=0.5)
axes[0].set_xlabel("T / K")
axes[0].set_ylabel("S / mK hPa$^{-1}$")
axes[1].scatter(predictors_annual["t_surf"].values, predictors_annual["max_d"].values, color='r', alpha=0.5)
axes[1].set_xlabel("T / K")
axes[1].set_ylabel("D / day$^{-1}$")
axes[2].scatter(predictors_annual["stability"].values, predictors_annual["max_d"].values, color='g', alpha=0.5)
axes[2].set_xlabel("S / mK hPa$^{-1}$")
axes[2].set_ylabel("D / day$^{-1}$")   

axes[1].plot(predictors_annual["t_surf"].values, predictors_annual["t_surf"].values * slopes_predictors_annual["max_d"] + intercepts_predictors_annual['max_d'], color='k')
axes[0].plot(predictors_annual["t_surf"].values, predictors_annual["t_surf"].values * slopes_predictors_annual["stability"] + intercepts_predictors_annual['stability'], color='k')
axes[2].plot(predictors_annual["stability"].values, predictors_annual["stability"].values * slopes_predictors_annual["max_d_stab"] + intercepts_predictors_annual['max_d_stab'], color='k')

axes[0].text(0.05, 0.95, f"$\partial S / \partial T = ${slopes_predictors_annual['stability']:.2e} mK hPa$^{{-1}}$ K$^{{-1}}$", transform=axes[0].transAxes, verticalalignment='top')
axes[1].text(0.05, 0.95, f"$\partial D / \partial T = ${slopes_predictors_annual['max_d']:.2e} day$^{{-1}}$ K$^{{-1}}$", transform=axes[1].transAxes, verticalalignment='top')
axes[2].text(0.05, 0.95, f"$\partial D / \partial S = ${slopes_predictors_annual['max_d_stab']:.2e} hPa day$^{{-1}}$ mK$^{{-1}}$", transform=axes[2].transAxes, verticalalignment='top')
fig.tight_layout()
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle('Annual Anomalies')
fig.savefig('plots/anvil_thinning/iris/predictor_scatter_annual.png', dpi=300, bbox_inches='tight')


# %% plot r_value as function of iwp for monthly and annual regressions
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
colors = {
    "t_surf": "k",
    "stability": "b",
    "max_d": "r",
}
labels = {
    "t_surf": "$\partial f / \partial T$",
    "stability": "$\partial f / \partial S$",
    "max_d": "$\partial f / \partial D$",
}

# monthly
for predictor_name in predictors.keys():
    axes[0].plot(r[predictor_name].sel(iwp=slice(1e-3, 20)).iwp, r[predictor_name].sel(iwp=slice(1e-3, 20)), label=labels[predictor_name], color=colors[predictor_name])
    axes[1].plot(r_annual[predictor_name].sel(iwp=slice(1e-3, 20)).iwp, r_annual[predictor_name].sel(iwp=slice(1e-3, 20)), label=labels[predictor_name], color=colors[predictor_name])
axes[0].set_xscale("log")
axes[0].set_title("Monthly regression")
axes[0].legend()
axes[1].set_title("Annual regression")
axes[0].set_ylabel("r-value")
for ax in axes:
    ax.set_xlabel('$I$ / kg m$^{-2}$')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig('plots/anvil_thinning/iris/r_values.png', dpi=300, bbox_inches='tight')

# %%
