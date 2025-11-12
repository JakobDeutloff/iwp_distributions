# %%
import os 

# %%
# lnsp, geop, sw-cs, lw-cs, temp
params = ['152', '129', '235003', '235004', '130']
sources = ['an', 'an', 'fc', 'fc', 'an']

for param, source in zip(params, sources):
    os.system(f"sbatch scripts/download/api_job_mars.sh {param} {source}")


