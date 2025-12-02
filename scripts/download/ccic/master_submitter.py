# %% 
import os
import numpy as np
# %%
years = [str(year) for year in range(2000, 2025)]
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] 
# %%
for year in years:
    os.system(f"sbatch scripts/download/ccic/submitter.sh {year}")
# %%
