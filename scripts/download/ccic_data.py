# %%
import s3fs
import os
from concurrent.futures import ThreadPoolExecutor

# %%
s3 = s3fs.S3FileSystem(anon=True)
prefix = 'chalmerscloudiceclimatology/record/cpcir/2008/ccic_cpcir_20080101'
files = s3.glob(prefix + '*')

# %%
local_dir = '/work/bm1183/m301049/ccic'
os.makedirs(local_dir, exist_ok=True)

def download_file(s3_path):
    local_path = os.path.join(local_dir, os.path.basename(s3_path))
    print(f"Downloading {s3_path} to {local_path}")
    s3.get(s3_path, local_path)

# Download in parallel (adjust max_workers as needed)
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(download_file, files)
# %%
