# %%
import matplotlib.pyplot as plt
import xarray as xr

# %% load data
hists_raw = {}
hists_weighted = {}
names = {"jed0011": "control", "jed0022": "plus4K", "jed0033": "plus2K"}

for run in names.keys():
    hists_raw[run] = xr.open_dataset(
        f"/work/bu1562/m301049/icon_hcap_data/{names[run]}/production/daily_cycle_hist_2d.nc"
    ).sum(["time", "local_time"]).coarsen(iwp=4, boundary="trim").sum()
    hists_weighted[run] = xr.open_dataset(
        f"/work/bu1562/m301049/icon_hcap_data/{names[run]}/production/daily_cycle_hist_weighted.nc"
    ).sum(["time", "local_time"]).coarsen(iwp=4, boundary="trim").sum()
# %% plot change in frequency of occurrence of iwp bins for each local time bin
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    hists_raw["jed0033"]["iwp"],
    (hists_raw["jed0033"]["hist"] / hists_raw["jed0033"]["size"])
    - (hists_raw["jed0011"]["hist"] / hists_raw["jed0011"]["size"]),
    label="raw",
)
ax.plot(
    hists_weighted["jed0033"]["iwp"],
    (hists_weighted["jed0033"]["hist"] / hists_weighted["jed0033"]["size"])
    - (hists_weighted["jed0011"]["hist"] / hists_weighted["jed0011"]["size"]),
    label="weighted",
)
ax.set_xscale("log")

# %% plot frequency of occurrence of iwp bins for each local time bin
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    hists_raw["jed0011"]["iwp"],
    (hists_raw["jed0011"]["hist"] / hists_raw["jed0011"]["size"]),
    label="raw",
)
ax.plot(
    hists_weighted["jed0011"]["iwp"],
    (hists_weighted["jed0011"]["hist"] / hists_weighted["jed0011"]["size"]),
    label="weighted",
)
ax.set_xscale("log")

# %%
