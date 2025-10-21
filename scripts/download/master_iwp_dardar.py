# %% 
import os 

# %%
years = list(range(2006, 2020))
batches = list(range(10))

# %%
for year in [2008, 2011]:
    for batch in batches:
        os.system(f'sbatch /home/m/m301049/iwp_distributions/scripts/download/submit_iwp_dardar.sh {str(year)} {str(batch)}')
# %%
