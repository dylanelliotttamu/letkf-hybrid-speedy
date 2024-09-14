import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import xarray as xr
import glob
from datetime import datetime, timedelta
from numba import jit

@jit()
def rms(true,prediction):
    return np.sqrt(np.nanmean((prediction-true)**2))

@jit()
def rms_tendency(variable,hours):
    variable_tendency = np.zeros((hours))
    variable = np.exp(variable) * 1000.0

    for i in range(hours):
        variable_tendency[i] = np.sqrt(np.mean((variable[i+1] - variable[i])**2.0))

    return variable_tendency


nature_file = '/scratch/user/troyarcomano/letkf-speedy/experiments/test/nature.nc'
analysis_file = '/scratch/user/troyarcomano/letkf-speedy/experiments/test/double/anal_mean.nc'
spread_file = '/scratch/user/troyarcomano/letkf-speedy/experiments/test/double/anal_sprd.nc'

ds_nature = xr.open_dataset(nature_file)
ds_analysis_mean = xr.open_dataset(analysis_file)
ds_spread = xr.open_dataset(spread_file)

level = 0.51

temp_500_nature = ds_nature['t'].sel(lev=level).values
temp_500_analysis = ds_analysis_mean['t'].sel(lev=level).values
temp_500_spread = ds_spread['t'].sel(lev=level).values

length = 84*6

analysis_rmse = np.zeros((length))
global_average_ensemble_spread= np.zeros((length))
print(np.shape(temp_500_nature))
for i in range(length):
    analysis_rmse[i] = rms(temp_500_nature[i,:,:],temp_500_analysis[i,:,:]) 
    global_average_ensemble_spread[i] = np.average(temp_500_spread[i,:,:])
    

x = np.arange(0,length)
#plt.plot(x,analysis_rmse,label='RMSE')
plt.plot(x,global_average_ensemble_spread,label='Ensemble Spread')
#plt.title('LETKF Analysis Error\nSigma Level 0.51 Temperature')
plt.xlabel('Cycle Number')
plt.ylabel('Ensemble Spread')
plt.show()
    
