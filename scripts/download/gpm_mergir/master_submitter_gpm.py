# %% 
import os 
import numpy as np 

# %%
years = np.arange(2000, 2026).astype(str)

# %%
for year in years[1:]:
    os.system(f'sbatch scripts/download/gpm_mergir/submitter_gpm.sh {year}')
# %%
