# %%
import os 
import numpy as np

# %% 
years = np.arange(1980, 2024, 1).astype(str)
for year in years:
    os.system(f"sbatch scripts/diagnose/gridsat/submitter_gridsat.sh {year}")