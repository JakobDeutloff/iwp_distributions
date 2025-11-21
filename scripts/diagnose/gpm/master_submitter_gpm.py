# %%
import os 
import numpy as np

# %% 
years = np.arange(2000, 2025, 1).astype(str)
for year in years:
    os.system(f"sbatch scripts/diagnose/gpm/submitter_gpm.sh {year}")