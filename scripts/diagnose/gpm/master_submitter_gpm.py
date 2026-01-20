# %%
import os 
import numpy as np

# %% 
years = np.arange(2000, 2025, 1).astype(str)
regions = ["all", "sea"]
for year in years:
    for region in regions:
        os.system(f"sbatch scripts/diagnose/gpm/submitter_gpm.sh {year} {region}")