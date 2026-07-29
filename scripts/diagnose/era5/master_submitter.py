# %% 
import os 
import numpy as np

# %% 
years = np.arange(1980, 2025, 1).astype(str)
regions = ["all", "sea"]
for year in years:
    for region in regions:
        os.system(f"sbatch scripts/diagnose/era5/submitter.sh {year} {region}")