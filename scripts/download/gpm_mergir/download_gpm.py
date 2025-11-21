# %%
import earthaccess
import sys

# %%
auth = earthaccess.login(strategy="netrc")
year = sys.argv[1]
next_year = str(int(year) + 1)

# Search for the granule
granules = earthaccess.search_data(
    short_name="GPM_MERGIR",
    version="1",
    cloud_hosted=True,
    temporal=(f"{year}-01-01T00:00:00Z", f"{next_year}-01-01T00:00:00Z")
)

print(len(granules), "granule(s) found")

#%% Download it
gr = earthaccess.download(granules, local_path='/work/bm1183/m301049/GPM_MERGIR/')




# %%
