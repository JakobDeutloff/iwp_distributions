# %% 
import os 

# %%
years = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
for year in years:
    os.system(f'sbatch /home/m/m301049/iwp_distributions/scripts/download/submit_iwp_dardar.sh {str(year)}')
# %%
