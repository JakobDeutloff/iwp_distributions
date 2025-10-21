good_years = ["2007", "2008", "2009", "2010", "2013", "2014", "2015", "2016"]
hists_2c_annual = hists_2c.resample(time="1YE").mean()
mask = np.isin(hists_2c_annual["time"].dt.year.astype(str), good_years)
mask_good_years = xr.DataArray(
    mask, dims=["time"], coords={"time": hists_2c_annual["time"]}
)
hists_2c_annual = hists_2c_annual.where(mask_good_years)
hists_annual["2c"] = (hists_2c_annual["hist"] / hists_2c_annual["size"]).transpose()

# %% regress annual histograms on annual mean temperature in every bin
from scipy.stats import linregress

slopes_ccic = []
err_ccic = []

temp_ccic = t_annual.sel(time=hists_annual["ccic"].time)
for i in range(hists_annual["ccic"].bin_center.size):
    hist_vals = hists_annual["ccic"].isel(bin_center=i).values
    res = linregress(temp_ccic.values, hist_vals)
    slopes_ccic.append(res.slope)
    err_ccic.append(res.stderr)

slopes_annual["ccic"] = xr.DataArray(
    slopes_ccic,
    coords={"bin_center": hists_annual["ccic"].bin_center},
    dims=["bin_center"],
)
error_annual["ccic"] = xr.DataArray(
    err_ccic,
    coords={"bin_center": hists_annual["ccic"].bin_center},
    dims=["bin_center"],
)


hists_dummy = hists_annual["2c"].where(mask_good_years, drop=True)
temp_2c = t_annual.sel(time=hists_dummy.time)
slopes_2c = []
err_2c = []
for i in range(hists_dummy.bin_center.size):
    hist_vals_2c = hists_dummy.isel(bin_center=i).values
    res_2c = linregress(temp_2c.values, hist_vals_2c)
    slopes_2c.append(res_2c.slope)
    err_2c.append(res_2c.stderr)

slopes_annual["2c"] = xr.DataArray(
    slopes_2c, coords={"bin_center": hists_annual["2c"].bin_center}, dims=["bin_center"]
)
error_annual["2c"] = xr.DataArray(
    err_2c, coords={"bin_center": hists_annual["2c"].bin_center}, dims=["bin_center"]
)
# %% plot annual variability and regression slopes
fig_ccic, axes_ccic = plot_regression(
    temp_ccic,
    hists_annual["ccic"],
    slopes_annual["ccic"],
    error_annual["ccic"],
    "CCIC Annual",
)
fig_ccic.savefig("plots/ccic_annual.png", dpi=300, bbox_inches="tight")

# %%
fig_2c, axes_2c = plot_regression(
    temp_2c,
    hists_annual["2c"],
    slopes_annual["2c"],
    error_annual["2c"],
    "2C-ICE Annual",
)
fig_2c.savefig("plots/2c_annual.png", dpi=300, bbox_inches="tight")