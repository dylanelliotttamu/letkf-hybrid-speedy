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

def latituded_weighted_rmse(true,prediction,lats):
    diff = prediction-true

    weights = np.cos(np.deg2rad(lats))

    weights2d = np.zeros(np.shape(diff))

    diff_squared = diff**2.0
    #weights = np.ones((10,96))

    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)

    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights2d)

    return np.sqrt(weighted_average)

start_year = 2011

startdate = datetime(2011,5,28,0)
enddate = datetime(2011,10,1,0)

nature_file = f'/skydata2/troyarcomano/ERA_5/{start_year}/era_5_y{start_year}_regridded_mpi_fixed_var.nc'
#analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_uniform_2011jan1_crash_april.nc'

#analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_crash_5_28_to_6_17_2011.nc'
analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/uniform_analysis_may282011_oct2011.nc'
#'/skydata2/troyarcomano/letkf-hybrid-speedy/DATA/era5_letkf_1990_2011/era5_letkf_1990_2011.nc'

analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_crash_5_28_to_6_17_2011.nc' #Dylan run
#analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/newest_test_letkf.nc' #may282011 Troy Run
#'/skydata2/troyarcomano/letkf-hybrid-speedy/experiments/hybrid_first_test/anal_mean.nc' #'/skydata2/troyarcomano/letkf-hybrid-speedy/DATA/ensemble/anal/mean/test.nc' #'~/stable_run/rtpp_0_3.nc' #'/skydata2/troyarcomano/letkf-hybrid-speedy/DATA/ensemble/anal/mean/test.nc'
spread_file =  '/skydata2/troyarcomano/letkf-hybrid-speedy/experiments/hybrid_first_test/anal_sprd.nc' #'/skydata2/troyarcomano/letkf-hybrid-speedy/DATA/ensemble/anal/mean/test.nc' #'~/stable_run/rtpp_0_3.nc' # '/skydata2/troyarcomano/letkf-hybrid-speedy/DATA/ensemble/anal/mean/test.nc'

ds_nature = xr.open_dataset(nature_file)
ds_analysis_mean = xr.open_dataset(analysis_file)
ds_analysis_mean_speedy = xr.open_dataset(analysis_file_speedy)
ds_spread = xr.open_dataset(spread_file)
###
def get_obs_era5_timeseries(startdate,enddate,timestep,lat_slice,lon_slice,sigma):
    start_year = startdate.year
    end_year = enddate.year

    currentdate = startdate
    counter = 0
    while currentdate.year <= enddate.year:
        print(currentdate.year)
        try:
           ds_nature = xr.open_dataset(f'/skydata2/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi.nc')
        except:
           ds_nature = xr.open_dataset(f'/skydata2/troyarcomano/ERA_5/{currentdate.year}/era_5_y{currentdate.year}_regridded_mpi_fixed_var.nc')

        begin_year = datetime(currentdate.year,1,1,0)
        begin_year_str = begin_year.strftime("%Y-%m-%d")
        attrs = {"units": f"hours since {begin_year_str} "}
        ds_nature = ds_nature.assign_coords({"Timestep": ("Timestep", ds_nature.Timestep.values, attrs)})
        ds_nature = xr.decode_cf(ds_nature)

        var = ['Specific_Humidity','logp']#var = ['U-wind','logp']
        ds_nature = ds_nature['Specific_Humidity'].sel(Lat=lat_slice,Lon=lon_slice,Sigma_Level=sigma)
        if start_year == currentdate.year:
           ds_merged = ds_nature
        else:
           ds_merged = xr.merge([ds_merged,ds_nature])

        currentdate = currentdate + timedelta(hours=ds_nature.sizes['Timestep'])

    time_slice = slice(startdate.strftime("%Y-%m-%d"),enddate.strftime("%Y-%m-%d"),timestep)
    return  ds_merged.sel(Timestep=time_slice)

lats = ds_nature.Lat
###
lons = ds_nature.Lon
###
level = 0.95 #0.2#0.95#0.51
level_era = 7 #2#7 #4

#era_time_points = 24*365
#era_timestep_6hr = era_time_points / 6
time_slice = slice(startdate,enddate)

var_era = 'Specific_Humidity'#'Temperature' #'V-wind'
var_da =  'q'#'t'#'q'
temp_500_nature = get_obs_era5_timeseries(datetime(2011,5,28,0),datetime(2011,10,1,0),1,lats,lons,level_era).values
#temp_500_nature = ds_nature[var_era].sel(Sigma_Level=level_era).values #NO TIME=time_slice???
temp_500_analysis = ds_analysis_mean[var_da].sel(lev=level).values
temp_500_analysis_speedy = ds_analysis_mean_speedy[var_da].sel(lev=level,time=time_slice).values
temp_500_spread = ds_spread[var_da].sel(lev=level).values

#print(np.info(temp_500_nature))
print(np.shape(temp_500_analysis_speedy))
ps_nature = ds_nature['logp'].values
ps_nature = 1000.0 * np.exp(ps_nature)
ps_analysis = ds_analysis_mean['ps'].values/100.0

xgrid = 96
ygrid = 48
length = 82#240  #1450 ##338 #160#64#177#1400#455

analysis_rmse = np.zeros((length))
analysis_rmse_speedy = np.zeros((length))
global_average_ensemble_spread= np.zeros((length))
ps_rmse = np.zeros((length))

analysis_error = np.zeros((length,ygrid,xgrid))
analysis_error_speedy = np.zeros((length,ygrid,xgrid))
###

#def get_obs_era5_timeseries(startdate,enddate,timestep,lat_slice,lon_slice,sigma):

#print(get_obs_era5_timeseries(datetime(2011,5,28,0),datetime(2011,10,1,0),6,lats,lons,level_era).values) # works! I think?
###
print(np.shape(temp_500_nature))
print(np.shape(temp_500_analysis))
for i in range(length):
    analysis_rmse[i] = latituded_weighted_rmse(temp_500_nature[i*6,:,:],temp_500_analysis[i,:,:],lats)
    analysis_rmse_speedy[i] = latituded_weighted_rmse(temp_500_nature[i*6,:,:],temp_500_analysis_speedy[i,:,:],lats)
    ps_rmse[i] = rms(ps_nature[i*6,:,:],ps_analysis[i,:,:])
    analysis_error[i,:,:] = temp_500_analysis[i,:,:] - temp_500_nature[i*6,:,:]
    analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - temp_500_nature[i*6,:,:]
    #global_average_ensemble_spread[i] = np.average(temp_500_spread[i,:,:])

#print(np.shape(temp_500_nature[i*6,:,:]))
''' 24(below) instead of 28 to cut transient event (ML spin up) out in first few weeks '''
  
averaged_error = np.average(abs(analysis_error[24::,:,:]),axis=0)
averaged_error_speedy = np.average(abs(analysis_error_speedy[24::,:,:]),axis=0)

lat = ds_analysis_mean.lat.values
lon = ds_analysis_mean.lon.values

lons2d, lats2d = np.meshgrid(lon,lat)

fig = plt.figure(figsize=(6,10))
''' ax1 ===>  Makes map of hybrid letkf analysis error  '''
ax1 = plt.subplot(311,projection=ccrs.PlateCarree())
ax1.coastlines()

''' Multiply averaged_error by 1000 for spec_humid only'''

cyclic_data, cyclic_lons = add_cyclic_point(averaged_error*1000, coord=lon)
lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

cf = ax1.contourf(lons2d, lats2d,cyclic_data,levels=np.arange(0,3.1,.05),extend='both')

plt.colorbar(cf,label='(g/kg)')
ax1.set_title('Hybrid LETKF Analysis Error\n Low Level Specific Humidity')

diff = averaged_error - averaged_error_speedy
cyclic_data, cyclic_lons = add_cyclic_point(diff, coord=lon)
lons2d,lats2d = np.meshgrid(cyclic_lons,lat)
'''ax2 ===>  makes plot of speedt letkf analysis error '''
ax2 = plt.subplot(312,projection=ccrs.PlateCarree())
ax2.coastlines()

cyclic_data, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

cf = ax2.contourf(lons2d, lats2d,cyclic_data,levels=np.arange(0,3.1,.05),extend='both')
plt.colorbar(cf,label='(g/kg)')
ax2.set_title('SPEEDY LETKF Analysis Error \n Low Level Specific Humidity')
''' Times 1000 on diff for Specific Humidity to be in g/kg'''
diff = (averaged_error - averaged_error_speedy)
cyclic_data, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

'''ax3 ==> Makes map of difference of hybrid and speedy '''

ax3 = plt.subplot(313,projection=ccrs.PlateCarree())
ax3.coastlines()
ax3.set_title('Difference (Hybrid - SPEEDY)')
cf = ax3.contourf(lons2d, lats2d,cyclic_data,levels=np.arange(-2,2.1,.025),extend='both',cmap='seismic')
plt.colorbar(cf,label='(g/kg)')
#plt.suptitle('Troy Ran Hybrid')
plt.show()


#print('Average RMSE Surface Pressure (hPa)',np.average(ps_rmse))
print('Average RMSE Variable',np.average(ps_rmse))
x = np.arange(0,length)

base = datetime(2011,5,28,0)

date_list = [base + timedelta(days=x/4) for x in range(length)]
plt.plot(date_list,analysis_rmse,color='r',linewidth=1,label='RMSE Hybrid ') #cov-infl1.3
plt.plot(date_list,analysis_rmse_speedy,color='b',linewidth=1,label='RMSE SPEEDY')
plt.axhline(y=np.average(analysis_rmse[:-1:]), color='r', linestyle='--',label="Average RMSE Hybrid")
plt.axhline(y=np.average(analysis_rmse_speedy[:-1:]), color='b', linestyle='--',label="Average RMSE SPEEDY")
print('average rmse Hybrid', np.average(analysis_rmse[20::]))
print('average rmse SPEEDY', np.average(analysis_rmse_speedy[20::]))
#plt.plot(date_list,global_average_ensemble_spread,label='Ensemble Spread')
#plt.title('LETKF Analysis Error\n Low Level Specific Humidity')
plt.title('Dylan Ran Hybrid\n LETKF Analysis Error\n Low Level Specific Humidity')
#plt.title('Ensemble Spread\nModel Level 4 Temperature')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Analysis Error')
#plt.ylabel('Analysis Error (kg/kg)')
#plt.ylabel('Ensemble Spread (K)')
plt.ylim(0,0.005)
plt.xlim([datetime(2011, 5, 27,0), datetime(2011, 6, 19,0)])
plt.show()
    
