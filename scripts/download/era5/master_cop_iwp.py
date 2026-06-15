# %%
import os 
import numpy as np

# %%
years = np.arange(1980, 2026)

# %%
for year in years:
    os.system(f"sbatch scripts/download/era5/api_job_copernicus.sh {str(year)}")
# %%
