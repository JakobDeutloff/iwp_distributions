# %% 
import os
import numpy as np
# %%
years = [str(year) for year in range(2000, 2025)]
# %%
for year in years:
    os.system(f"sbatch scripts/download/ccic/submitter.sh {year}")
# %%
