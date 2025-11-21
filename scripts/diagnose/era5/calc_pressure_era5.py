# %% 
import xarray as xr
import pandas as pd
import numpy as np

# %%
lnsp = xr.open_dataarray('/work/bm1183/m301049/era5/monthly/152_an_latlon.grb', engine='cfgrib').load()
coeffs = pd.read_csv('/work/bm1183/m301049/era5/monthly/coeff_pressure_era5.csv', index_col=0, delimiter=';')
# %%
coeffs_xr = xr.Dataset(
    {
        'a': (('hybrid', 'latitude', 'longitude'), np.tile(coeffs['a'].values[:, np.newaxis, np.newaxis], (1, len(lnsp['latitude']), len(lnsp['longitude'])))),
        'b': (('hybrid', 'latitude', 'longitude'), np.tile(coeffs['b'].values[:, np.newaxis, np.newaxis], (1, len(lnsp['latitude']), len(lnsp['longitude'])))),
    },
    coords={
        'hybrid': coeffs.index.values,  
        'latitude': lnsp['latitude'],
        'longitude': lnsp['longitude'],
    }
)
# %% 
p_s = np.exp(lnsp)
p = coeffs_xr['a'] + coeffs_xr['b'] * p_s

# %%
p = p.transpose('time', 'hybrid', 'latitude', 'longitude')
p.attrs = {
    'long_name': 'Pressure',
    'units': 'hPa',
}
p.to_netcdf('/work/bm1183/m301049/era5/monthly/pressure_latlon.nc')

# %%
