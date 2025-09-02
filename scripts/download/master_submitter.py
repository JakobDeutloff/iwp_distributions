# %% 
import os
import numpy as np
# %%
years = np.arange(1998, 2024)
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] 
# %%
for year in years:
    for month in months:
        os.system(f"sbatch scripts/download/submitter.sh {year} {month}")
# %%
