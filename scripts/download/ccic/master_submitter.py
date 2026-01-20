# %% 
import os
import numpy as np
# %%
years = [str(year) for year in range(2000, 2025)]
regions = ["all", "sea"]
# %%
for year in years:
    for region in regions:
        os.system(f"sbatch scripts/download/ccic/submitter.sh {year} {region}")