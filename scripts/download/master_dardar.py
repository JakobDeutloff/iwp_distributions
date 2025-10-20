# %%
import subprocess
import time

# %% 
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

def submit_and_wait(script, year):
    # Submit job and get job ID
    result = subprocess.run(['sbatch', script, year], capture_output=True, text=True)
    job_id = None
    for part in result.stdout.split():
        if part.isdigit():
            job_id = part
            break
    if job_id is None:
        raise RuntimeError("Could not get job ID from sbatch output.")

    # Wait for job to finish
    while True:
        squeue = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
        if job_id not in squeue.stdout:
            break
        # wait for 1 min before checking again
        time.sleep(60)  

for year in years:
    submit_and_wait('/home/m/m301049/iwp_distributions/scripts/download/download_dardar.sh', str(year))
    submit_and_wait('/home/m/m301049/iwp_distributions/scripts/download/submit_processing_dardar.sh', str(year))
# %%
