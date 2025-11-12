# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from dask.diagnostics import ProgressBar
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress
from src.plot import plot_regression, plot_hists, definitions
from src.helper_functions import nan_detrend, interpolate_bins
import xrft


# %% initialize containers
hists_monthly = {}
hists_smooth = {}
slopes_monthly = {}
error_montly = {}
bins = bins = np.logspace(-3, 2, 254)[::4]
datasets = ["ccic", "2c", "dardar", "spare"]
colors, line_labels, linestyles = definitions()

# %% open CCIC
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

hists_ccic = xr.concat(hist_list, dim="time")

# %% open 2C-ICE
hists_2c = xr.open_dataset("/work/bm1183/m301049/cloudsat/dists.nc")

# %% open dardar
hists_dardar = xr.open_dataset("/work/bm1183/m301049/dardarv3.10/hist_dardar.nc")

# %% open spareice
hists_spare = xr.open_dataset("/work/bm1183/m301049/spareice/hists_metop.nc")

# %% coarsen histograms and normalise by size
hists_ccic_coarse = hists_ccic.coarsen(bin_center=4, boundary="trim").sum()

# ccic
hists_monthly["ccic"] = (
    hists_ccic_coarse["hist"].resample(time="1ME").sum()
    / hists_ccic_coarse["size"].resample(time="1ME").sum()
)

# 2c-ice
hists_monthly["2c"] = (hists_2c["hist"] / hists_2c["size"]).sel(
    time=slice("2006-06", "2017-12")
)

# dardar
hists_monthly["dardar"] = (
    hists_dardar["hist"].resample(time="1ME").sum()
    / hists_dardar["size"].resample(time="1ME").sum()
).sel(time=slice("2006-06", "2017-12"))
hists_monthly["dardar"] = hists_monthly["dardar"].where(
    hists_dardar["size"].resample(time="1ME").sum() > 2.1e6
)

# spareice
hists_monthly["spare"] = hists_spare["hist"] / hists_spare["size"]
hists_monthly["spare"] = hists_monthly["spare"].transpose("bin_center", "time")
hists_monthly["spare"] = hists_monthly["spare"].sel(time=slice(None, "2025-07"))

# %% fix time coordinates
for key in datasets:
    hists_monthly[key]["time"] = pd.to_datetime(
        hists_monthly[key]["time"].dt.strftime("%Y-%m")
    )

# %% load era5 surface temp
path_t2m = "/pool/data/ERA5/E5/sf/an/1M/167/"
# List all .grb files
files = glob.glob(path_t2m + "E5sf00_1M_*.grb")

# Filter files for year > 2000
files_after_2000 = [
    f for f in files if int(re.search(r"_(\d{4})_", f).group(1)) >= 2000
]
ds = xr.open_mfdataset(files_after_2000, engine="cfgrib", combine="by_coords")

#  slect tropics and calculate annual average
with ProgressBar():
    t_month = (
        ds["t2m"]
        .where((ds["latitude"] >= -30) & (ds["latitude"] <= 30))
        .mean("values")
        .compute()
    )

t_month["time"] = pd.to_datetime(t_month["time"].dt.strftime("%Y-%m"))
t_annual = t_month.resample(time="1YE").mean("time")

# %% detrend
# temperature
t_detrend = xr.DataArray(
    detrend(t_month), coords={"time": t_month["time"]}, dims=t_month.dims
)
hists_detrend = {}
for ds in datasets:
    hists_detrend[ds] = nan_detrend(hists_monthly[ds])

t_deseason = t_detrend.groupby("time.month") - t_detrend.groupby("time.month").mean(
    "time"
)
# %% fft analysis
fft_spectra = {}

# FFT
dt = 1
N = t_detrend.size
x = t_deseason.values
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1)

# Keep only positive frequencies
mask = freqs > 0
psd = (2.0 * np.abs(X) ** 2) / (N**2 * dt)
period = 1 / freqs[mask]
power_da = xr.DataArray(psd[mask], coords={"period": period}, dims=["period"])
# %% plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(power_da["period"], power_da, label="ERA5 T2M", color="blue")
ax.set_xlabel("Period (months)")



# %%
