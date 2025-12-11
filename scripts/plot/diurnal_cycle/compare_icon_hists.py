# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# %% load icon
runs = ["jed0011", "jed0022", "jed0033"]
names = {
    "jed0011": "control",
    "jed0022": "plus4K",
    "jed0033": "plus2K",
}
temp_delta = {
    "jed0011": 0,
    "jed0022": 4,
    "jed0033": 2,
}
colors = {
    "jed0011": "black",
    "jed0022": "red",
    "jed0033": "blue",
}
hists_1d = {}
hists_2d = {}
for run in runs:
    hists_1d[run] = xr.open_dataarray(
        f"/work/bm1183/m301049/icon_hcap_data/publication/distributions/{run}_deep_clouds_daily_cycle.nc"
    ).sum('day').rename({'local_hour': 'local_time'})
    hists_2d[run] = (
        xr.open_dataset(
            f"/work/bm1183/m301049/icon_hcap_data/{names[run]}/production/daily_cycle_hist_2d.nc"
        )
        .sum("time")
    )
    hists_1d[run]['local_time'] = hists_2d[run]['local_time']

# %% check if absolute number of samples is similar 
for run in runs:
    total_samples_1d = hists_1d[run].sum().values
    total_samples_2d = hists_2d[run]["hist"].sel(iwp=slice(1e0, None)).sum().values
    print(
        f"Run {run} ({names[run]}): Total samples 1D = {total_samples_1d}, Total samples 2D = {total_samples_2d}"
    )

# %% calculate 1d hists from 2d hists 
hists_1d_from_2d = {}
for run in runs:
    hist_2d = hists_2d[run]["hist"]
    hists_1d_from_2d[run] = hist_2d.sel(iwp=slice(1e0, None)).sum("iwp")

# %% plot normalised hists 
fig, ax = plt.subplots(figsize=(8, 6))
for run in runs:
    norm_hist_1d = hists_1d[run] / hists_1d[run].sum()
    norm_hist_2d = hists_1d_from_2d[run] / hists_1d_from_2d[run].sum()
    ax.plot(
        norm_hist_1d["local_time"],
        norm_hist_1d,
        label=f"1D {names[run]}",
        linestyle='-',
        color=colors[run],
    )
    ax.plot(
        norm_hist_2d["local_time"],
        norm_hist_2d,
        label=f"2D {names[run]}",
        linestyle='--',
        color=colors[run],
    )

# %%
