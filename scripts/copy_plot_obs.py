import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from netCDF4 import Dataset
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
import xarray as xr
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import *
from numba import jit
import calendar
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns

obs_network_file = '/skydata2/troyarcomano/letkf-hybrid-speedy/obs/networks/uniform.txt' #uniform.txt'#real.txt'
analysis_file = '/skydata2/troyarcomano/letkf-hybrid-speedy/experiments/ERA_5_full/double/anal_mean.nc'

network_ij = np.loadtxt(obs_network_file,skiprows=2,dtype=int)
network_ij = network_ij - 1
print(network_ij[1])
ds_analysis_mean = xr.open_dataset(analysis_file)

lat = ds_analysis_mean.lat.values
lon = ds_analysis_mean.lon.values

print(lon[network_ij[1][0]],lat[network_ij[1][1]])
lons2d, lats2d = np.meshgrid(lon,lat)



plt.figure(figsize=(16,6))
ax1 = plt.subplot(111,projection=ccrs.PlateCarree())
ax1.coastlines()

for i in range(np.shape(lon)[0]):
    plt.axvline(lon[i]-180,linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

for i in range(np.shape(lat)[0]):
    plt.axhline(lat[i],linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

for i in range(np.shape(network_ij)[0]):
    plt.scatter(lon[network_ij[i][0]],lat[network_ij[i][1]],color='k')

plt.title("Uniform Observation Network")
plt.tight_layout()
plt.savefig("uniform_network.pdf",dpi=1200)
plt.show()    




